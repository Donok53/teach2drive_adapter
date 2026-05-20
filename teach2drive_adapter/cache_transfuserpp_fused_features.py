import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Teach2DriveIndexDataset
from .transfuserpp_bridge import load_transfuserpp, prepare_transfuserpp_inputs


def _camera_list(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def _capture_backbone_fused_features(net):
    captured = {}

    def hook(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            if len(output) >= 3:
                captured["fused"] = output[1]
            elif len(output) >= 1:
                captured["fused"] = output[0]
        else:
            captured["fused"] = output

    handle = net.backbone.register_forward_hook(hook)
    return captured, handle


def _write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_cache(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    cameras = _camera_list(args.cameras)
    indices = None
    if args.max_samples > 0:
        indices = np.arange(args.max_samples, dtype=np.int64)
    dataset = Teach2DriveIndexDataset(
        args.index,
        indices=indices,
        cameras=cameras,
        image_size=tuple(args.image_size),
        lidar_size=args.lidar_size,
        episode_root_override=args.episode_root_override,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    net, config, load_info = load_transfuserpp(args.garage_root, args.team_config, device=device, checkpoint=args.checkpoint)
    net.eval()
    captured, handle = _capture_backbone_fused_features(net)

    feature_store = None
    sample_index = np.lib.format.open_memmap(out_dir / "sample_index.npy", mode="w+", dtype=np.int64, shape=(len(dataset),))
    sample_episode = np.lib.format.open_memmap(out_dir / "sample_episode.npy", mode="w+", dtype=np.int64, shape=(len(dataset),))
    sample_frame = np.lib.format.open_memmap(out_dir / "sample_frame.npy", mode="w+", dtype=np.int64, shape=(len(dataset),))
    metadata = {
        "index": str(Path(args.index).expanduser()),
        "episode_root_override": args.episode_root_override,
        "garage_root": str(Path(args.garage_root).expanduser()),
        "team_config": str(Path(args.team_config).expanduser()),
        "checkpoint": args.checkpoint,
        "cameras": cameras,
        "tfpp_camera": args.tfpp_camera,
        "command_mode": args.command_mode,
        "image_size": list(args.image_size),
        "lidar_size": int(args.lidar_size),
        "dtype": args.dtype,
        "samples": int(len(dataset)),
        "load_info": load_info,
    }
    _write_metadata(out_dir / "metadata.json", metadata)

    start = time.time()
    cursor = 0
    try:
        for step, batch in enumerate(loader, start=1):
            scalar = batch["scalar"].to(device, non_blocking=True)
            camera = batch["camera"].to(device, non_blocking=True)
            lidar = batch["lidar"].to(device, non_blocking=True)
            inputs = prepare_transfuserpp_inputs(
                scalar=scalar,
                camera=camera,
                lidar=lidar,
                cameras=cameras,
                config=config,
                command_mode=args.command_mode,
                tfpp_camera=args.tfpp_camera,
            )
            captured.clear()
            with torch.no_grad():
                _ = net(**inputs)
            if "fused" not in captured:
                raise RuntimeError("Backbone hook did not capture fused features")
            fused = captured["fused"].detach().cpu()
            if fused.ndim != 4:
                raise ValueError(f"Expected fused features [B,C,H,W], got {tuple(fused.shape)}")
            batch_size = int(fused.shape[0])
            if feature_store is None:
                feature_shape = tuple(int(v) for v in fused.shape[1:])
                feature_dtype = np.float16 if args.dtype == "float16" else np.float32
                feature_store = np.lib.format.open_memmap(
                    out_dir / "fused_features.npy",
                    mode="w+",
                    dtype=feature_dtype,
                    shape=(len(dataset), *feature_shape),
                )
                metadata["feature_shape"] = list(feature_shape)
                _write_metadata(out_dir / "metadata.json", metadata)
            end = cursor + batch_size
            feature_store[cursor:end] = fused.numpy().astype(feature_store.dtype, copy=False)
            sample_index[cursor:end] = batch["index"].numpy().astype(np.int64, copy=False)
            sample_episode[cursor:end] = batch["episode_idx"].numpy().astype(np.int64, copy=False)
            sample_frame[cursor:end] = batch["frame_idx"].numpy().astype(np.int64, copy=False)
            cursor = end
            if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
                elapsed = max(time.time() - start, 1e-6)
                print(f"feature_cache_step={step:05d}/{len(loader):05d} samples={cursor} samples/s={cursor/elapsed:.1f}", flush=True)
    finally:
        handle.remove()
    if cursor != len(dataset):
        raise RuntimeError(f"Feature cache wrote {cursor} samples, expected {len(dataset)}")
    metadata["samples"] = int(cursor)
    _write_metadata(out_dir / "metadata.json", metadata)
    print(json.dumps({"output_dir": str(out_dir), **metadata}, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache frozen TransFuser++ fused backbone features.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--garage-root", required=True)
    parser.add_argument("--team-config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--episode-root-override", default="")
    parser.add_argument("--cameras", default="front,left,right")
    parser.add_argument("--tfpp-camera", default="front")
    parser.add_argument("--command-mode", choices=["lane_follow", "target_angle"], default="target_angle")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--lidar-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    build_cache(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
