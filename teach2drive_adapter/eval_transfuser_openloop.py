import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .data import Teach2DriveIndexDataset
from .transfuser_bridge import batch_from_teach2drive_sample


def _install_torch_scatter_stub() -> None:
    if importlib.util.find_spec("torch_scatter") is not None:
        return
    module = types.ModuleType("torch_scatter")

    def _missing(*_args, **_kwargs):
        raise RuntimeError("torch_scatter is required only when TransFuser use_point_pillars=True.")

    module.scatter_mean = _missing
    module.scatter_max = _missing
    sys.modules["torch_scatter"] = module


def _load_transfuser_modules(transfuser_root: Path):
    team_dir = transfuser_root / "team_code_transfuser"
    if not team_dir.exists():
        raise FileNotFoundError(f"Could not find team_code_transfuser at {team_dir}")
    _install_torch_scatter_stub()
    # TransFuser 2022 predates NumPy's removal of these aliases.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "object"):
        np.object = object  # type: ignore[attr-defined]
    sys.path.insert(0, str(team_dir))
    from config import GlobalConfig  # type: ignore
    from data import draw_target_point  # type: ignore
    from model import LidarCenterNet  # type: ignore

    return GlobalConfig, LidarCenterNet, draw_target_point


def _load_agent_args(team_config: Path) -> Dict:
    args_path = team_config / "args.txt"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing TransFuser args.txt: {args_path}")
    return json.loads(args_path.read_text(encoding="utf-8"))


def _configure(GlobalConfig, agent_args: Dict):
    config = GlobalConfig(setting="eval")
    for key in ("sync_batch_norm", "use_point_pillars", "n_layer", "use_target_point_image"):
        if key in agent_args:
            setattr(config, key, agent_args[key])
    return config


def _load_net(transfuser_root: Path, team_config: Path, device: torch.device):
    GlobalConfig, LidarCenterNet, draw_target_point = _load_transfuser_modules(transfuser_root)
    agent_args = _load_agent_args(team_config)
    config = _configure(GlobalConfig, agent_args)
    backbone = agent_args.get("backbone", "transFuser")
    image_architecture = agent_args.get("image_architecture", "resnet34")
    lidar_architecture = agent_args.get("lidar_architecture", "resnet18")
    use_velocity = bool(agent_args.get("use_velocity", True))
    if bool(getattr(config, "use_point_pillars", False)):
        raise RuntimeError("This evaluator uses bridged BEV tensors and does not support point-pillars checkpoints yet.")

    net = LidarCenterNet(config, str(device), backbone, image_architecture, lidar_architecture, use_velocity)
    if bool(agent_args.get("sync_batch_norm", False)):
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    ckpts = sorted(team_config.glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No .pth files found in {team_config}")
    state = torch.load(ckpts[0], map_location=device)
    if all(str(key).startswith("module.") for key in state.keys()):
        state = {key[7:]: value for key, value in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net, config, draw_target_point, {"checkpoint": str(ckpts[0]), "missing": len(missing), "unexpected": len(unexpected), "args": agent_args}


def _target_point_image(draw_target_point, target_point: torch.Tensor, device: torch.device) -> torch.Tensor:
    images: List[torch.Tensor] = []
    for item in target_point.detach().cpu().numpy():
        image = draw_target_point(item.astype(np.float32))
        images.append(torch.from_numpy(image).to(device=device, dtype=torch.float32))
    return torch.stack(images, dim=0)


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    net, config, draw_target_point, load_info = _load_net(Path(args.transfuser_root).expanduser(), Path(args.team_config).expanduser(), device)
    dataset = Teach2DriveIndexDataset(
        args.index,
        cameras=[item.strip() for item in args.cameras.split(",") if item.strip()],
        image_size=tuple(args.image_size),
        lidar_size=args.lidar_size,
        episode_root_override=args.episode_root_override,
    )
    if args.sample_stride > 1:
        sample_ids = list(range(0, len(dataset), args.sample_stride))
        if args.max_samples > 0:
            sample_ids = sample_ids[: args.max_samples]
    elif args.max_samples > 0 and args.max_samples < len(dataset):
        sample_ids = np.linspace(0, len(dataset) - 1, args.max_samples, dtype=np.int64).astype(int).tolist()
    else:
        sample_ids = list(range(len(dataset)))

    errors_vehicle = []
    errors_lidar = []
    per_sample = []
    for n, sample_id in enumerate(sample_ids, start=1):
        sample = dataset[sample_id]
        tf_batch = batch_from_teach2drive_sample(sample)
        rgb = tf_batch["rgb"].to(device=device, dtype=torch.float32)
        lidar = tf_batch["lidar"].to(device=device, dtype=torch.float32)
        target_point = tf_batch["target_point"].to(device=device, dtype=torch.float32)
        velocity = tf_batch["velocity"].to(device=device, dtype=torch.float32)
        target_point_image = _target_point_image(draw_target_point, target_point, device)

        with torch.no_grad():
            pred_wp, _ = net.forward_ego(rgb, lidar, target_point, target_point_image, velocity, num_points=None)

        pred = pred_wp.detach().cpu().numpy()[0].astype(np.float32)
        pred_vehicle = pred.copy()
        pred_vehicle[:, 0] += float(config.lidar_pos[0])
        target = sample["target"].numpy()[:12].reshape(4, 3)[:, :2].astype(np.float32)
        err_vehicle = np.linalg.norm(pred_vehicle[:, :2] - target, axis=1)
        err_lidar = np.linalg.norm(pred[:, :2] - target, axis=1)
        errors_vehicle.append(err_vehicle)
        errors_lidar.append(err_lidar)
        if len(per_sample) < args.save_samples:
            per_sample.append(
                {
                    "sample_id": int(sample_id),
                    "target_xy": target.astype(float).tolist(),
                    "pred_vehicle_xy": pred_vehicle[:, :2].astype(float).tolist(),
                    "pred_lidar_xy": pred[:, :2].astype(float).tolist(),
                    "err_vehicle_m": err_vehicle.astype(float).tolist(),
                    "err_lidar_m": err_lidar.astype(float).tolist(),
                }
            )
        if args.log_every > 0 and (n == 1 or n % args.log_every == 0):
            print(f"sample={n}/{len(sample_ids)} mean_xy_vehicle={np.mean(np.stack(errors_vehicle), axis=0).tolist()}", flush=True)

    metrics = {
        "index": str(Path(args.index).expanduser()),
        "transfuser_root": str(Path(args.transfuser_root).expanduser()),
        "team_config": str(Path(args.team_config).expanduser()),
        "device": str(device),
        "samples": len(sample_ids),
        "load_info": load_info,
        "mean_xy_error_vehicle_m_by_horizon": np.mean(np.stack(errors_vehicle), axis=0).astype(float).tolist(),
        "mean_xy_error_lidar_m_by_horizon": np.mean(np.stack(errors_lidar), axis=0).astype(float).tolist(),
        "per_sample_preview": per_sample,
        "notes": [
            "This is zero-shot open-loop evaluation before fitting a Teach2Drive adapter.",
            "LiDAR uses a Teach2Drive BEV approximation because raw point clouds are not stored in the current dataset.",
            "Vehicle-coordinate error adds TransFuser lidar_pos[0] back to predicted x before comparison.",
        ],
    }
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot open-loop check for official TransFuser weights on a Teach2Drive index.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--transfuser-root", default="/home/byeongjae/code/transfuser")
    parser.add_argument("--team-config", required=True, help="Folder containing TransFuser args.txt and .pth.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episode-root-override", default="")
    parser.add_argument("--cameras", default="left,front,right", help="Comma-separated Teach2Drive camera tokens. Use front for front-only simulation.")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--lidar-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--save-samples", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    evaluate(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
