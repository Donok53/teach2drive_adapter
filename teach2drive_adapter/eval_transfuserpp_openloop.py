import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import Teach2DriveIndexDataset
from .transfuserpp_bridge import base_target_from_checkpoint, load_transfuserpp, prepare_transfuserpp_inputs


def _camera_list(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    cameras = _camera_list(args.cameras)
    net, config, load_info = load_transfuserpp(args.garage_root, args.team_config, device=device, checkpoint=args.checkpoint)
    dataset = Teach2DriveIndexDataset(
        args.index,
        cameras=cameras,
        image_size=tuple(args.image_size),
        lidar_size=args.lidar_size,
        episode_root_override=args.episode_root_override,
    )
    if args.sample_stride > 1:
        sample_ids = list(range(0, len(dataset), args.sample_stride))
    else:
        sample_ids = list(range(len(dataset)))
    if args.max_samples > 0:
        sample_ids = sample_ids[: args.max_samples]

    xy_errors = []
    per_sample = []
    for n, sample_id in enumerate(sample_ids, start=1):
        sample = dataset[sample_id]
        scalar = sample["scalar"].unsqueeze(0).to(device=device, dtype=torch.float32)
        camera = sample["camera"].unsqueeze(0).to(device=device, dtype=torch.float32)
        lidar = sample["lidar"].unsqueeze(0).to(device=device, dtype=torch.float32)
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
            base_target = base_target_from_checkpoint(outputs[2], outputs[1], scalar, config, target_dim=17, speed_dim=4)
        pred_xy = base_target[:, :12].reshape(1, 4, 3)[0, :, :2].detach().cpu().numpy()
        target_xy = sample["target"][:12].reshape(4, 3)[:, :2].numpy()
        err = np.linalg.norm(pred_xy - target_xy, axis=1)
        xy_errors.append(err)
        if len(per_sample) < args.save_samples:
            per_sample.append(
                {
                    "sample_id": int(sample_id),
                    "target_xy": target_xy.astype(float).tolist(),
                    "pred_xy": pred_xy.astype(float).tolist(),
                    "err_m": err.astype(float).tolist(),
                }
            )
        if args.log_every > 0 and (n == 1 or n % args.log_every == 0):
            print(f"sample={n}/{len(sample_ids)} mean_xy={np.mean(np.stack(xy_errors), axis=0).tolist()}", flush=True)

    metrics = {
        "index": str(Path(args.index).expanduser()),
        "garage_root": str(Path(args.garage_root).expanduser()),
        "team_config": str(Path(args.team_config).expanduser()),
        "episode_root_override": args.episode_root_override,
        "device": str(device),
        "samples": len(sample_ids),
        "cameras": cameras,
        "tfpp_camera": args.tfpp_camera,
        "command_mode": args.command_mode,
        "load_info": load_info,
        "mean_xy_error_m_by_horizon": np.mean(np.stack(xy_errors), axis=0).astype(float).tolist(),
        "per_sample_preview": per_sample,
        "notes": [
            "This is a zero-shot open-loop diagnostic for frozen TransFuser++ before adapter fitting.",
            "TransFuser++ predicts route checkpoints, so the first four checkpoints are compared to Teach2Drive horizons as a coarse sanity check.",
            "The current Teach2Drive dataset stores BEV arrays rather than raw point clouds, so LiDAR input is an approximation.",
        ],
    }
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot open-loop diagnostic for CARLA Garage TransFuser++ on Teach2Drive data.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--garage-root", required=True)
    parser.add_argument("--team-config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episode-root-override", default="")
    parser.add_argument("--cameras", default="front,left,right")
    parser.add_argument("--tfpp-camera", default="front")
    parser.add_argument("--command-mode", choices=["lane_follow", "target_angle"], default="lane_follow")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--lidar-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--sample-stride", type=int, default=20)
    parser.add_argument("--save-samples", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    evaluate(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

