from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .vehicle_dynamics import BoundedYawRateCalibrator, calibrator_checkpoint_payload, checkpoint_curvature_phase


def _load_frames(path: Path) -> dict[str, np.ndarray] | None:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    if len(rows) < 3:
        return None
    x = np.asarray([float(row.get("odom", {}).get("x", 0.0)) for row in rows], dtype=np.float64)
    y = np.asarray([float(row.get("odom", {}).get("y", 0.0)) for row in rows], dtype=np.float64)
    yaw = np.unwrap(np.asarray([float(row.get("odom", {}).get("yaw", 0.0)) for row in rows], dtype=np.float64))
    cumulative = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    return {"x": x, "y": y, "yaw": yaw, "cumulative": cumulative}


def _expert_curvature_at_distance(frames: dict[str, np.ndarray], frame: int, distance: float) -> float | None:
    x, y, yaw = frames["x"], frames["y"], frames["yaw"]
    if frame < 0 or frame >= len(x) - 1:
        return None
    cumulative = frames["cumulative"]
    target_distance = cumulative[frame] + float(distance)
    upper = int(np.searchsorted(cumulative, target_distance, side="left"))
    if upper <= frame or upper >= len(x):
        return None
    lower = upper - 1
    width = cumulative[upper] - cumulative[lower]
    if width <= 1e-5:
        return None
    alpha = float((target_distance - cumulative[lower]) / width)
    target_x = x[lower] + alpha * (x[upper] - x[lower])
    target_y = y[lower] + alpha * (y[upper] - y[lower])
    dx, dy = target_x - x[frame], target_y - y[frame]
    lateral = -math.sin(yaw[frame]) * dx + math.cos(yaw[frame]) * dy
    return 2.0 * lateral / max(float(distance) ** 2, 1e-6)


def _build_samples(args: argparse.Namespace) -> dict[str, np.ndarray]:
    cache = np.load(Path(args.cache).expanduser(), allow_pickle=True)
    index = np.load(Path(args.index).expanduser(), allow_pickle=True)
    episode = cache["sample_episode"].astype(np.int64)
    frame = cache["sample_frame"].astype(np.int64)
    speed = cache["scalar"][:, 0].astype(np.float32)
    checkpoints = cache["checkpoint_flat"].astype(np.float32).reshape(-1, 10, 2)
    episode_dirs = [Path(str(value)).name for value in index["episode_dirs"]]
    root = Path(args.episode_root).expanduser()
    output: dict[str, list[float]] = {
        key: [] for key in ("episode", "frame", "speed", "curvature", "expert_curvature", "entry", "exit", "target_gain")
    }
    for episode_id in np.unique(episode):
        if int(episode_id) >= len(episode_dirs):
            continue
        frames = _load_frames(root / episode_dirs[int(episode_id)] / "frames.jsonl")
        if frames is None:
            continue
        for sample_id in np.flatnonzero(episode == episode_id):
            sample_speed = float(speed[sample_id])
            phase = checkpoint_curvature_phase(checkpoints[sample_id], args.phase_curvature_scale)
            curvature = float(phase["curvature"])
            lookahead = float(np.linalg.norm(checkpoints[sample_id, -1]))
            if sample_speed < args.minimum_speed or abs(curvature) < args.minimum_abs_curvature:
                continue
            if not args.minimum_lookahead <= lookahead <= args.maximum_lookahead:
                continue
            expert = _expert_curvature_at_distance(frames, int(frame[sample_id]), lookahead)
            if expert is None or abs(expert) > args.maximum_abs_expert_curvature:
                continue
            target_gain = float(np.clip(expert / curvature, args.minimum_gain, args.maximum_gain))
            # Strong exit phases are handed back to the frozen PID at runtime;
            # keep their calibration target at identity to avoid an unused,
            # poorly constrained extrapolation affecting partial exits.
            if phase["exit"] >= args.full_exit_strength:
                target_gain = 1.0
            values = {
                "episode": int(episode_id),
                "frame": int(frame[sample_id]),
                "speed": sample_speed,
                "curvature": curvature,
                "expert_curvature": expert,
                "entry": float(phase["entry"]),
                "exit": float(phase["exit"]),
                "target_gain": target_gain,
            }
            for key, value in values.items():
                output[key].append(value)
    return {key: np.asarray(value) for key, value in output.items()}


def _split(episode: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * val_ratio)))
    val_episodes = set(int(value) for value in episodes[:val_count])
    val = np.asarray([int(value) in val_episodes for value in episode], dtype=bool)
    return ~val, val


def _tensor(data: dict[str, np.ndarray], key: str, mask: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(data[key][mask], dtype=torch.float32)


def _metrics(model: BoundedYawRateCalibrator, data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    with torch.no_grad():
        gain = model(
            _tensor(data, "speed", mask),
            _tensor(data, "curvature", mask),
            _tensor(data, "entry", mask),
            _tensor(data, "exit", mask),
        ).cpu().numpy()
    curvature = data["curvature"][mask]
    expert = data["expert_curvature"][mask]
    entry = data["entry"][mask] >= 0.5
    exit_phase = data["exit"][mask] >= 0.5
    apex = ~(entry | exit_phase)
    result = {
        "count": int(mask.sum()),
        "base_curvature_mae": float(np.mean(np.abs(expert - curvature))),
        "calibrated_curvature_mae": float(np.mean(np.abs(expert - curvature * gain))),
        "gain_mean": float(gain.mean()),
        "gain_p05": float(np.quantile(gain, 0.05)),
        "gain_p50": float(np.quantile(gain, 0.50)),
        "gain_p95": float(np.quantile(gain, 0.95)),
        "identity_deviation": float(np.mean(np.abs(gain - 1.0))),
    }
    for phase_mask, name in ((entry, "entry"), (apex, "apex"), (exit_phase, "exit")):
        result[f"{name}_count"] = int(phase_mask.sum())
        result[f"{name}_gain_mean"] = float(gain[phase_mask].mean()) if np.any(phase_mask) else math.nan
        result[f"{name}_mae"] = (
            float(np.mean(np.abs(expert[phase_mask] - curvature[phase_mask] * gain[phase_mask])))
            if np.any(phase_mask)
            else math.nan
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a bounded geometric desired-yaw-rate calibrator")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.75)
    parser.add_argument("--maximum-gain", type=float, default=1.15)
    parser.add_argument("--minimum-speed", type=float, default=1.0)
    parser.add_argument("--minimum-abs-curvature", type=float, default=0.005)
    parser.add_argument("--maximum-abs-expert-curvature", type=float, default=0.5)
    parser.add_argument("--minimum-lookahead", type=float, default=2.0)
    parser.add_argument("--maximum-lookahead", type=float, default=20.0)
    parser.add_argument("--phase-curvature-scale", type=float, default=0.02)
    parser.add_argument("--full-exit-strength", type=float, default=0.75)
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--turn-weight", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=91)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = _build_samples(args)
    if len(data["episode"]) < 100:
        raise RuntimeError(f"Only {len(data['episode'])} valid curvature samples were built")
    train_mask, val_mask = _split(data["episode"], args.val_ratio, args.seed)
    model = BoundedYawRateCalibrator(args.minimum_gain, args.maximum_gain)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    speed = _tensor(data, "speed", train_mask)
    curvature = _tensor(data, "curvature", train_mask)
    entry = _tensor(data, "entry", train_mask)
    exit_phase = _tensor(data, "exit", train_mask)
    target = _tensor(data, "target_gain", train_mask)
    weight = 1.0 + args.turn_weight * (curvature.abs() / 0.10).clamp(0.0, 1.0)
    best_state, best_score, best_epoch = None, float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        gain = model(speed, curvature, entry, exit_phase)
        fit = (weight * nn.functional.smooth_l1_loss(gain, target, reduction="none", beta=0.05)).mean()
        identity = (gain - 1.0).square().mean()
        loss = fit + args.identity_weight * identity
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            metrics = _metrics(model, data, val_mask)
            score = metrics["calibrated_curvature_mae"] + 0.02 * metrics["identity_deviation"]
            print(
                f"epoch={epoch:03d} loss={loss.item():.6f} fit={fit.item():.6f} identity={identity.item():.6f} "
                f"val_mae={metrics['calibrated_curvature_mae']:.6f} gain={metrics['gain_mean']:.4f}",
                flush=True,
            )
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    train_metrics = _metrics(model, data, train_mask)
    val_metrics = _metrics(model, data, val_mask)
    print("train", json.dumps(train_metrics, sort_keys=True), flush=True)
    print("validation", json.dumps(val_metrics, sort_keys=True), flush=True)
    print("coefficients", dict(zip(model.feature_names, model.coefficients.detach().cpu().tolist())), flush=True)

    output = Path(args.out_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "feature_names": list(model.feature_names),
        "minimum_gain": args.minimum_gain,
        "maximum_gain": args.maximum_gain,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "args": vars(args),
    }
    torch.save(calibrator_checkpoint_payload(model, metadata), output / "best_model.pt")
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(output / "curvature_samples.npz", **data, train_mask=train_mask, val_mask=val_mask)
    print(f"saved={output / 'best_model.pt'}", flush=True)


if __name__ == "__main__":
    main()
