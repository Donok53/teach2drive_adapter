import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from .data import Teach2DriveIndexDataset, split_by_episode
from .transfuser_adapter_model import TransFuserResidualAdapterPolicy


COLORS = {
    "gt": (40, 220, 40),
    "base": (50, 120, 255),
    "adapter": (255, 200, 40),
    "axis": (85, 85, 85),
    "text": (245, 245, 245),
    "dark": (18, 18, 18),
}


def _camera_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_model(args: argparse.Namespace, dataset: Teach2DriveIndexDataset, device: torch.device) -> TransFuserResidualAdapterPolicy:
    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    model = TransFuserResidualAdapterPolicy(
        transfuser_root=args.transfuser_root or ckpt_args.get("transfuser_root", "/home/byeongjae/code/transfuser"),
        team_config=args.team_config or ckpt_args.get("team_config", "/home/byeongjae/code/transfuser/model_ckpt/models_2022/transfuser"),
        device=device,
        scalar_dim=dataset.scalar_dim,
        target_dim=dataset.target_dim + 1,
        hidden_dim=int(ckpt_args.get("hidden_dim", args.hidden_dim)),
        speed_dim=dataset.speed_dim,
    )
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    print(f"loaded={args.checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.to(device)
    model.eval()
    return model


def _predict(model: TransFuserResidualAdapterPolicy, sample: Dict[str, torch.Tensor], device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scalar = sample["scalar"].unsqueeze(0).to(device)
    camera = sample["camera"].unsqueeze(0).to(device)
    lidar = sample["lidar"].unsqueeze(0).to(device)
    with torch.no_grad():
        rgb = model._camera_to_transfuser_rgb(camera)
        lidar_bev = model._lidar_to_transfuser_bev(lidar)
        target_point = model._target_point_from_scalar(scalar)
        velocity = scalar[:, :1].contiguous()
        _features, pred_wp_lidar = model._frozen_transfuser(rgb, lidar_bev, target_point, velocity)
        base = model._base_target_from_transfuser(pred_wp_lidar, scalar)
        adapted = model(scalar, camera, lidar)["target"]

    target = sample["target"].detach().cpu().numpy()
    traj_dim = adapted.shape[1] - model.speed_dim - 1
    gt_xy = target[:traj_dim].reshape(-1, 3)[:, :2]
    base_xy = base.detach().cpu().numpy()[0, :traj_dim].reshape(-1, 3)[:, :2]
    adapter_xy = adapted.detach().cpu().numpy()[0, :traj_dim].reshape(-1, 3)[:, :2]
    return gt_xy.astype(np.float32), base_xy.astype(np.float32), adapter_xy.astype(np.float32)


def _mean_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred - target, axis=1).mean())


def _front_image(sample: Dict[str, torch.Tensor], cameras: List[str]) -> np.ndarray:
    camera = sample["camera"].detach().cpu().numpy()
    front_idx = cameras.index("front") if "front" in cameras else min(0, camera.shape[0] - 1)
    image = np.transpose(camera[front_idx], (1, 2, 0))
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def _to_px(point: np.ndarray, size: int, scale: float) -> Tuple[int, int]:
    # Teach2Drive uses ego-frame x/y. Draw x forward/up and y lateral/right.
    x, y = float(point[0]), float(point[1])
    px = int(size * 0.5 + y * scale)
    py = int(size * 0.82 - x * scale)
    return px, py


def _draw_polyline(panel: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], scale: float, thickness: int = 3) -> None:
    pts = [_to_px(point, panel.shape[0], scale) for point in points]
    for point in pts:
        cv2.circle(panel, point, 5, color, -1, cv2.LINE_AA)
    if len(pts) >= 2:
        cv2.polylines(panel, [np.asarray(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def _trajectory_panel(gt_xy: np.ndarray, base_xy: np.ndarray, adapter_xy: np.ndarray, base_err: float, adapter_err: float, size: int = 520) -> np.ndarray:
    panel = np.full((size, size, 3), COLORS["dark"], dtype=np.uint8)
    scale = size / 18.0
    for meters in range(2, 18, 2):
        y = int(size * 0.82 - meters * scale)
        cv2.line(panel, (0, y), (size, y), COLORS["axis"], 1, cv2.LINE_AA)
        cv2.putText(panel, f"{meters}m", (8, max(y - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["axis"], 1, cv2.LINE_AA)
    cv2.line(panel, (size // 2, 0), (size // 2, size), COLORS["axis"], 1, cv2.LINE_AA)
    ego = (size // 2, int(size * 0.82))
    cv2.circle(panel, ego, 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(panel, "ego", (ego[0] + 10, ego[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["text"], 1, cv2.LINE_AA)

    _draw_polyline(panel, gt_xy, COLORS["gt"], scale, 4)
    _draw_polyline(panel, base_xy, COLORS["base"], scale, 3)
    _draw_polyline(panel, adapter_xy, COLORS["adapter"], scale, 3)

    rows = [
        ("GT", COLORS["gt"]),
        (f"TransFuser {base_err:.2f}m", COLORS["base"]),
        (f"Adapter {adapter_err:.2f}m", COLORS["adapter"]),
        (f"gain {(base_err - adapter_err):+.2f}m", COLORS["text"]),
    ]
    y = 28
    for text, color in rows:
        cv2.putText(panel, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 28
    return panel


def _compose_frame(sample_id: int, sample: Dict[str, torch.Tensor], cameras: List[str], gt_xy: np.ndarray, base_xy: np.ndarray, adapter_xy: np.ndarray) -> np.ndarray:
    base_err = _mean_error(base_xy, gt_xy)
    adapter_err = _mean_error(adapter_xy, gt_xy)
    front = _front_image(sample, cameras)
    front = cv2.resize(front, (720, 405), interpolation=cv2.INTER_AREA)
    front = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
    cv2.rectangle(front, (0, 0), (720, 54), (0, 0, 0), -1)
    cv2.putText(front, f"sample {sample_id} | mean XY: TransFuser {base_err:.2f}m -> Adapter {adapter_err:.2f}m", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    traj = _trajectory_panel(gt_xy, base_xy, adapter_xy, base_err, adapter_err, size=520)
    pad_h = max(front.shape[0], traj.shape[0])
    canvas = np.full((pad_h, front.shape[1] + traj.shape[1], 3), (22, 22, 22), dtype=np.uint8)
    canvas[: front.shape[0], : front.shape[1]] = front
    canvas[: traj.shape[0], front.shape[1] :] = traj
    return canvas


def _select_indices(args: argparse.Namespace) -> List[int]:
    _train_idx, val_idx = split_by_episode(args.index, val_ratio=args.val_ratio, seed=args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(val_idx), generator=rng).numpy()
    val_idx = val_idx[perm]
    if args.max_eval_samples > 0:
        val_idx = val_idx[: args.max_eval_samples]
    return [int(item) for item in val_idx]


def visualize(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    cameras = _camera_list(args.cameras)
    sample_indices = _select_indices(args)
    dataset = Teach2DriveIndexDataset(
        args.index,
        indices=np.asarray(sample_indices, dtype=np.int64),
        cameras=cameras,
        image_size=tuple(args.image_size),
        lidar_size=args.lidar_size,
        episode_root_override=args.episode_root_override,
    )
    model = _load_model(args, dataset, device)

    records = []
    frames = []
    for local_idx in range(len(dataset)):
        sample = dataset[local_idx]
        gt_xy, base_xy, adapter_xy = _predict(model, sample, device)
        base_err = _mean_error(base_xy, gt_xy)
        adapter_err = _mean_error(adapter_xy, gt_xy)
        records.append(
            {
                "local_idx": local_idx,
                "sample_index": sample_indices[local_idx],
                "base_err": base_err,
                "adapter_err": adapter_err,
                "improvement": base_err - adapter_err,
            }
        )
        frames.append((sample, gt_xy, base_xy, adapter_xy))
        if (local_idx + 1) % max(args.log_every, 1) == 0:
            print(f"evaluated {local_idx + 1}/{len(dataset)}", flush=True)

    if args.selection == "top":
        chosen = sorted(records, key=lambda row: row["improvement"], reverse=True)[: args.num_frames]
    elif args.selection == "worst":
        chosen = sorted(records, key=lambda row: row["improvement"])[: args.num_frames]
    else:
        step = max(len(records) // max(args.num_frames, 1), 1)
        chosen = records[::step][: args.num_frames]

    rendered = []
    for rank, row in enumerate(chosen):
        sample, gt_xy, base_xy, adapter_xy = frames[row["local_idx"]]
        frame = _compose_frame(row["sample_index"], sample, cameras, gt_xy, base_xy, adapter_xy)
        rendered.append(frame)
        image_path = out_dir / f"{rank:03d}_sample_{row['sample_index']}.jpg"
        cv2.imwrite(str(image_path), frame)

    if rendered:
        grid_cols = int(args.grid_cols)
        grid_rows = int(np.ceil(len(rendered) / grid_cols))
        h, w = rendered[0].shape[:2]
        grid = np.full((grid_rows * h, grid_cols * w, 3), (18, 18, 18), dtype=np.uint8)
        for idx, frame in enumerate(rendered):
            r, c = divmod(idx, grid_cols)
            grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame
        cv2.imwrite(str(out_dir / "comparison_grid.jpg"), grid)

        video_path = out_dir / "comparison.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {video_path}")
        for frame in rendered:
            for _ in range(max(int(args.hold_frames), 1)):
                writer.write(frame)
        writer.release()

    summary = {
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "samples_evaluated": len(records),
        "selection": args.selection,
        "selected": chosen,
        "mean_base_err": float(np.mean([row["base_err"] for row in records])),
        "mean_adapter_err": float(np.mean([row["adapter_err"] for row in records])),
        "mean_improvement": float(np.mean([row["improvement"] for row in records])),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize TransFuser zero-shot vs Teach2Drive residual adapter trajectories.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--transfuser-root", default="/home/byeongjae/code/transfuser")
    parser.add_argument("--team-config", default="/home/byeongjae/code/transfuser/model_ckpt/models_2022/transfuser")
    parser.add_argument("--episode-root-override", default="")
    parser.add_argument("--cameras", default="left,front,right")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--lidar-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-eval-samples", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--selection", choices=["top", "uniform", "worst"], default="top")
    parser.add_argument("--grid-cols", type=int, default=2)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--hold-frames", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    visualize(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
