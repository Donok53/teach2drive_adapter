from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .vehicle_dynamics_v4 import (
    HorizonYawRateCalibrator,
    calibrator_checkpoint_payload,
    checkpoint_pure_pursuit_sequence,
)


def _load_episode(path: Path) -> dict[str, np.ndarray] | None:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    if len(rows) < 4:
        return None
    return {
        "x": np.asarray([float(row.get("odom", {}).get("x", 0.0)) for row in rows]),
        "y": np.asarray([float(row.get("odom", {}).get("y", 0.0)) for row in rows]),
        "yaw": np.unwrap(np.asarray([float(row.get("odom", {}).get("yaw", 0.0)) for row in rows])),
    }


def _expert_curvature(episode: dict[str, np.ndarray], frame: int, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, yaw = episode["x"], episode["y"], episode["yaw"]
    values = np.zeros(len(distances), dtype=np.float64)
    valid = np.zeros(len(distances), dtype=bool)
    if frame < 0 or frame >= len(x) - 1:
        return values, valid
    cumulative = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    for horizon, distance in enumerate(distances):
        target_distance = cumulative[frame] + float(distance)
        upper = int(np.searchsorted(cumulative, target_distance, side="left"))
        if upper <= frame or upper >= len(x):
            continue
        lower = upper - 1
        width = float(cumulative[upper] - cumulative[lower])
        if width <= 1e-5:
            continue
        alpha = float((target_distance - cumulative[lower]) / width)
        tx = float(x[lower] + alpha * (x[upper] - x[lower]))
        ty = float(y[lower] + alpha * (y[upper] - y[lower]))
        dx, dy = tx - float(x[frame]), ty - float(y[frame])
        lateral = -math.sin(float(yaw[frame])) * dx + math.cos(float(yaw[frame])) * dy
        values[horizon] = 2.0 * lateral / max(float(distance) ** 2, 1e-6)
        valid[horizon] = True
    return values, valid


def _build_samples(args: argparse.Namespace) -> dict[str, np.ndarray]:
    cache = np.load(Path(args.cache).expanduser(), allow_pickle=True)
    index = np.load(Path(args.index).expanduser(), allow_pickle=True)
    sample_episode = cache["sample_episode"].astype(np.int64)
    sample_frame = cache["sample_frame"].astype(np.int64)
    speed = cache["scalar"][:, 0].astype(np.float64)
    checkpoints = cache["checkpoint_flat"].astype(np.float64).reshape(-1, 10, 2)
    episode_dirs = [Path(str(value)).name for value in index["episode_dirs"]]
    root = Path(args.episode_root).expanduser()
    output: dict[str, list] = {key: [] for key in ("episode", "speed", "curvature", "expert_curvature", "entry", "exit", "target_gain", "mask")}
    horizons = tuple(float(value) for value in args.horizons)
    for episode_id in np.unique(sample_episode):
        if int(episode_id) >= len(episode_dirs):
            continue
        episode = _load_episode(root / episode_dirs[int(episode_id)] / "frames.jsonl")
        if episode is None:
            continue
        for sample_id in np.flatnonzero(sample_episode == episode_id):
            sample_speed = float(speed[sample_id])
            if sample_speed < args.minimum_speed:
                continue
            target = checkpoint_pure_pursuit_sequence(
                checkpoints[sample_id], sample_speed, horizons, phase_curvature_scale=args.phase_curvature_scale
            )
            curvature = np.asarray(target["curvature"], dtype=np.float64)
            distances = np.asarray(target["sample_distance"], dtype=np.float64)
            expert, valid = _expert_curvature(episode, int(sample_frame[sample_id]), distances)
            valid &= np.abs(curvature) >= args.minimum_abs_curvature
            valid &= np.abs(expert) <= args.maximum_abs_expert_curvature
            valid &= curvature * expert > 0.0
            if not np.any(valid):
                continue
            gain = np.ones_like(curvature)
            gain[valid] = np.clip(expert[valid] / curvature[valid], args.minimum_gain, args.maximum_gain)
            if float(target["exit"]) >= args.full_exit_strength:
                gain[valid] = 1.0
            output["episode"].append(int(episode_id))
            output["speed"].append(sample_speed)
            output["curvature"].append(curvature)
            output["expert_curvature"].append(expert)
            output["entry"].append(float(target["entry"]))
            output["exit"].append(float(target["exit"]))
            output["target_gain"].append(gain)
            output["mask"].append(valid.astype(np.float32))
    return {key: np.asarray(value) for key, value in output.items()}


def _split(episodes: np.ndarray, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    count = max(1, int(round(len(unique) * ratio)))
    selected = set(int(value) for value in unique[:count])
    validation = np.asarray([int(value) in selected for value in episodes])
    return ~validation, validation


def _predict(model: HorizonYawRateCalibrator, data: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(
            torch.as_tensor(data["speed"][rows], dtype=torch.float32),
            torch.as_tensor(data["curvature"][rows], dtype=torch.float32),
            torch.as_tensor(data["entry"][rows], dtype=torch.float32),
            torch.as_tensor(data["exit"][rows], dtype=torch.float32),
        ).cpu().numpy()


def _metrics(model: HorizonYawRateCalibrator, data: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, object]:
    gain = _predict(model, data, rows)
    curvature = data["curvature"][rows]
    expert = data["expert_curvature"][rows]
    mask = data["mask"][rows].astype(bool)
    base_error = np.abs(expert - curvature)
    calibrated_error = np.abs(expert - curvature * gain)
    return {
        "sample_count": int(rows.sum()),
        "valid_target_count": int(mask.sum()),
        "base_curvature_mae": float(base_error[mask].mean()),
        "calibrated_curvature_mae": float(calibrated_error[mask].mean()),
        "improvement_fraction": float(1.0 - calibrated_error[mask].mean() / max(base_error[mask].mean(), 1e-12)),
        "gain_mean_by_horizon": [float(gain[:, index][mask[:, index]].mean()) for index in range(gain.shape[1])],
        "identity_deviation": float(np.abs(gain[mask] - 1.0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit bounded future-pose yaw targets for dynamics v4")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--horizons", type=float, nargs=3, default=(0.25, 0.50, 0.75))
    parser.add_argument("--minimum-gain", type=float, default=0.85)
    parser.add_argument("--maximum-gain", type=float, default=1.10)
    parser.add_argument("--minimum-speed", type=float, default=1.0)
    parser.add_argument("--minimum-abs-curvature", type=float, default=0.005)
    parser.add_argument("--maximum-abs-expert-curvature", type=float, default=0.5)
    parser.add_argument("--phase-curvature-scale", type=float, default=0.02)
    parser.add_argument("--full-exit-strength", type=float, default=0.75)
    parser.add_argument("--identity-weight", type=float, default=2.0)
    parser.add_argument("--turn-weight", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=91)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = _build_samples(args)
    if len(data["episode"]) < 100:
        raise RuntimeError(f"Only {len(data['episode'])} valid samples were built")
    train_rows, val_rows = _split(data["episode"], args.val_ratio, args.seed)
    model = HorizonYawRateCalibrator(3, args.minimum_gain, args.maximum_gain)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    speed = torch.as_tensor(data["speed"][train_rows], dtype=torch.float32)
    curvature = torch.as_tensor(data["curvature"][train_rows], dtype=torch.float32)
    entry = torch.as_tensor(data["entry"][train_rows], dtype=torch.float32)
    exit_phase = torch.as_tensor(data["exit"][train_rows], dtype=torch.float32)
    target = torch.as_tensor(data["target_gain"][train_rows], dtype=torch.float32)
    mask = torch.as_tensor(data["mask"][train_rows], dtype=torch.float32)
    weight = mask * (1.0 + args.turn_weight * (curvature.abs() / 0.10).clamp(0.0, 1.0))
    best_state, best_score, best_epoch = None, float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        gain = model(speed, curvature, entry, exit_phase)
        fit_element = nn.functional.smooth_l1_loss(gain, target, reduction="none", beta=0.05)
        fit = (weight * fit_element).sum() / weight.sum().clamp_min(1.0)
        identity = (mask * (gain - 1.0).square()).sum() / mask.sum().clamp_min(1.0)
        loss = fit + args.identity_weight * identity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            metrics = _metrics(model, data, val_rows)
            score = float(metrics["calibrated_curvature_mae"]) + 0.02 * float(metrics["identity_deviation"])
            print(f"epoch={epoch:03d} loss={loss.item():.6f} val_mae={metrics['calibrated_curvature_mae']:.6f} gains={metrics['gain_mean_by_horizon']}", flush=True)
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    train_metrics = _metrics(model, data, train_rows)
    val_metrics = _metrics(model, data, val_rows)
    metadata = {
        "horizons_seconds": list(args.horizons),
        "feature_names": list(model.feature_names),
        "minimum_gain": args.minimum_gain,
        "maximum_gain": args.maximum_gain,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "args": vars(args),
    }
    output = Path(args.out_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(calibrator_checkpoint_payload(model, metadata), output / "best_model.pt")
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(output / "curvature_samples.npz", **data, train_rows=train_rows, val_rows=val_rows)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print(f"saved={output / 'best_model.pt'}", flush=True)


if __name__ == "__main__":
    main()
