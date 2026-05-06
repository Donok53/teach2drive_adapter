import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import Teach2DriveIndexDataset
from .transfuserpp_bridge import base_target_from_checkpoint, load_transfuserpp, prepare_transfuserpp_inputs, speed_expectation


def _camera_list(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def _flatten_checkpoint(pred_checkpoint: torch.Tensor, width: int) -> torch.Tensor:
    flat = torch.zeros((pred_checkpoint.shape[0], width), dtype=pred_checkpoint.dtype, device=pred_checkpoint.device)
    raw = pred_checkpoint.reshape(pred_checkpoint.shape[0], -1)
    flat[:, : min(width, raw.shape[1])] = raw[:, : flat.shape[1]]
    return flat


def build_cache(args: argparse.Namespace) -> None:
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    net, config, load_info = load_transfuserpp(args.garage_root, args.team_config, device=device, checkpoint=args.checkpoint)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs for prior caching", flush=True)
        net = nn.DataParallel(net)
    net.eval()

    chunks = {
        "sample_index": [],
        "sample_episode": [],
        "sample_frame": [],
        "scalar": [],
        "layout": [],
        "target": [],
        "stop_state": [],
        "stop_reason": [],
        "stop_reason_mask": [],
        "sample_weight": [],
        "base_target": [],
        "checkpoint_flat": [],
        "speed_logits": [],
        "expected_speed": [],
    }
    checkpoint_width = int(getattr(config, "predict_checkpoint_len", 10)) * 2
    speed_classes = len(getattr(config, "target_speeds", []))
    start = time.time()
    seen = 0
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
        with torch.no_grad():
            outputs = net(**inputs)
            pred_target_speed = outputs[1]
            pred_checkpoint = outputs[2]
            base_target = base_target_from_checkpoint(
                pred_checkpoint=pred_checkpoint,
                pred_target_speed=pred_target_speed,
                scalar=scalar,
                config=config,
                target_dim=batch["target"].shape[1],
                speed_dim=args.speed_dim,
            )
            if pred_checkpoint is None:
                checkpoint_flat = torch.zeros((scalar.shape[0], checkpoint_width), dtype=scalar.dtype, device=device)
            else:
                checkpoint_flat = _flatten_checkpoint(pred_checkpoint, checkpoint_width)
            if pred_target_speed is None:
                speed_logits = torch.zeros((scalar.shape[0], speed_classes), dtype=scalar.dtype, device=device)
            else:
                speed_logits = pred_target_speed
            expected_speed = speed_expectation(pred_target_speed, config, scalar.shape[0], device)

        chunks["sample_index"].append(batch["index"].numpy().astype(np.int64))
        chunks["sample_episode"].append(batch["episode_idx"].numpy().astype(np.int64))
        chunks["sample_frame"].append(batch["frame_idx"].numpy().astype(np.int64))
        chunks["scalar"].append(batch["scalar"].numpy().astype(np.float32))
        chunks["layout"].append(batch["layout"].numpy().astype(np.float32))
        chunks["target"].append(batch["target"].numpy().astype(np.float32))
        chunks["stop_state"].append(batch["stop_state"].numpy().astype(np.int64))
        chunks["stop_reason"].append(batch["stop_reason"].numpy().astype(np.int64))
        chunks["stop_reason_mask"].append(batch["stop_reason_mask"].numpy().astype(np.float32))
        chunks["sample_weight"].append(batch["sample_weight"].numpy().astype(np.float32))
        chunks["base_target"].append(base_target.detach().cpu().numpy().astype(np.float32))
        chunks["checkpoint_flat"].append(checkpoint_flat.detach().cpu().numpy().astype(np.float32))
        chunks["speed_logits"].append(speed_logits.detach().cpu().numpy().astype(np.float32))
        chunks["expected_speed"].append(expected_speed.detach().cpu().numpy().astype(np.float32))

        seen += int(scalar.shape[0])
        if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
            elapsed = max(time.time() - start, 1e-6)
            print(f"cache_step={step:05d}/{len(loader):05d} samples={seen} samples/s={seen/elapsed:.1f}", flush=True)

    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    metadata = {
        "index": str(Path(args.index).expanduser()),
        "episode_root_override": args.episode_root_override,
        "garage_root": str(Path(args.garage_root).expanduser()),
        "team_config": str(Path(args.team_config).expanduser()),
        "cameras": cameras,
        "tfpp_camera": args.tfpp_camera,
        "command_mode": args.command_mode,
        "samples": int(len(arrays["sample_index"])),
        "load_info": load_info,
        "target_speeds": list(getattr(config, "target_speeds", [])),
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata), dtype=object)
    np.savez(out_path, **arrays)
    (out_path.with_suffix(".json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), **metadata}, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache frozen TransFuser++ priors for fast Teach2Drive adapter training.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--garage-root", required=True)
    parser.add_argument("--team-config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--episode-root-override", default="")
    parser.add_argument("--cameras", default="front,left,right")
    parser.add_argument("--tfpp-camera", default="front")
    parser.add_argument("--command-mode", choices=["lane_follow", "target_angle"], default="target_angle")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--lidar-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--speed-dim", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    build_cache(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

