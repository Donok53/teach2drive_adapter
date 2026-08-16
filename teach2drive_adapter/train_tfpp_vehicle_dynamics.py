from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .vehicle_dynamics import SpeedConditionedYawDynamics, checkpoint_payload


def _load_episode(path: Path, smooth_frames: int) -> dict[str, np.ndarray] | None:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    if len(rows) < 4:
        return None
    time = np.asarray([float(row.get("time", 0.0)) for row in rows], dtype=np.float64)
    yaw = np.unwrap(np.asarray([float(row.get("odom", {}).get("yaw", 0.0)) for row in rows], dtype=np.float64))
    speed = np.asarray([float(row.get("odom", {}).get("v_forward", 0.0)) for row in rows], dtype=np.float64)
    steer = np.asarray([float(row.get("control", {}).get("steer", 0.0)) for row in rows], dtype=np.float64)
    brake = np.asarray([float(row.get("control", {}).get("brake", 0.0)) for row in rows], dtype=np.float64)
    dt = np.diff(time)
    if not np.any(dt > 1e-4):
        return None
    raw_rate = np.zeros_like(yaw)
    raw_rate[1:] = np.diff(yaw) / np.maximum(dt, 1e-3)
    raw_rate[0] = raw_rate[1]
    width = max(int(smooth_frames), 1)
    yaw_rate = np.asarray([raw_rate[max(0, i - width + 1) : i + 1].mean() for i in range(len(raw_rate))])
    return {"speed": speed, "steer": steer, "brake": brake, "yaw_rate": yaw_rate, "dt": dt}


def _features(speed: np.ndarray, yaw_rate: np.ndarray, steer: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            yaw_rate,
            (speed / 20.0) * yaw_rate,
            np.ones_like(speed),
            speed * steer,
            (speed**2 / 20.0) * steer,
            steer,
            steer**3,
        )
    )


def _pack(episodes: list[dict[str, np.ndarray]], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, targets, steering = [], [], []
    for episode in episodes:
        v, s, b, r = (episode[key] for key in ("speed", "steer", "brake", "yaw_rate"))
        mask = (
            (v[:-1] >= args.minimum_speed)
            & (b[:-1] < args.maximum_brake)
            & (np.abs(s[:-1]) <= args.maximum_abs_steer)
            & (np.abs(r[:-1]) <= args.maximum_abs_yaw_rate)
            & (np.abs(r[1:]) <= args.maximum_abs_yaw_rate)
        )
        features.append(_features(v[:-1][mask], r[:-1][mask], s[:-1][mask]))
        targets.append(r[1:][mask])
        steering.append(s[:-1][mask])
    return np.concatenate(features), np.concatenate(targets), np.concatenate(steering)


def _robust_weighted_ridge(x: np.ndarray, y: np.ndarray, steer: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    turn = np.clip(np.abs(steer) / max(args.turn_scale, 1e-6), 0.0, 1.0)
    base_weight = 1.0 + args.turn_weight * turn
    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    eye = np.eye(x.shape[1], dtype=np.float64)
    eye[2, 2] = 0.0
    robust = np.ones_like(y)
    for _ in range(max(args.irls_iterations, 1)):
        weight = base_weight * robust
        root = np.sqrt(weight)
        xw, yw = x * root[:, None], y * root
        coefficients = np.linalg.solve(xw.T @ xw + args.ridge * eye, xw.T @ yw)
        residual = y - x @ coefficients
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-4)
        robust = np.minimum(1.0, args.huber_delta * scale / np.maximum(np.abs(residual), 1e-8))
    return coefficients


def _metrics(name: str, x: np.ndarray, y: np.ndarray, steer: np.ndarray, coefficients: np.ndarray) -> dict[str, float]:
    prediction = x @ coefficients
    persistence = x[:, 0]
    state_x = x[:, :3]
    state_c = np.linalg.solve(state_x.T @ state_x + 1e-3 * np.eye(3), state_x.T @ y)
    state_prediction = state_x @ state_c
    turn = np.abs(steer) >= 0.05
    result = {
        "count": int(len(y)),
        "mae": float(np.mean(np.abs(y - prediction))),
        "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "r2": float(1.0 - np.sum((y - prediction) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)),
        "persistence_mae": float(np.mean(np.abs(y - persistence))),
        "state_only_mae": float(np.mean(np.abs(y - state_prediction))),
        "turn_mae": float(np.mean(np.abs(y[turn] - prediction[turn]))) if np.any(turn) else math.nan,
        "turn_count": int(turn.sum()),
    }
    print(name, json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit planner-independent Tesla steer-to-yaw-rate dynamics")
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--smooth-frames", type=int, default=3)
    parser.add_argument("--minimum-speed", type=float, default=1.0)
    parser.add_argument("--maximum-brake", type=float, default=0.5)
    parser.add_argument("--maximum-abs-steer", type=float, default=0.95)
    parser.add_argument("--maximum-abs-yaw-rate", type=float, default=1.5)
    parser.add_argument("--turn-scale", type=float, default=0.20)
    parser.add_argument("--turn-weight", type=float, default=4.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--irls-iterations", type=int, default=5)
    parser.add_argument("--huber-delta", type=float, default=1.5)
    args = parser.parse_args()

    paths = sorted(Path(args.episode_root).expanduser().glob("**/frames.jsonl"))
    loaded = [(path, _load_episode(path, args.smooth_frames)) for path in paths]
    loaded = [(path, data) for path, data in loaded if data is not None]
    if len(loaded) < 2:
        raise RuntimeError(f"Need at least two valid episodes under {args.episode_root}")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(loaded))
    val_count = max(1, int(round(len(loaded) * args.val_ratio)))
    val_ids = set(int(index) for index in order[:val_count])
    train_episodes = [data for index, (_, data) in enumerate(loaded) if index not in val_ids]
    val_episodes = [data for index, (_, data) in enumerate(loaded) if index in val_ids]
    x_train, y_train, s_train = _pack(train_episodes, args)
    x_val, y_val, s_val = _pack(val_episodes, args)
    coefficients = _robust_weighted_ridge(x_train, y_train, s_train, args)
    train_metrics = _metrics("train", x_train, y_train, s_train, coefficients)
    val_metrics = _metrics("validation", x_val, y_val, s_val, coefficients)
    print("coefficients", dict(zip(SpeedConditionedYawDynamics.feature_names, coefficients.tolist())), flush=True)

    output = Path(args.out_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    model = SpeedConditionedYawDynamics(coefficients)
    metadata = {
        "feature_names": list(model.feature_names),
        "coefficients": coefficients.tolist(),
        "step_seconds": float(np.median(np.concatenate([data["dt"] for _, data in loaded]))),
        "smooth_frames": args.smooth_frames,
        "minimum_speed": args.minimum_speed,
        "episode_count": len(loaded),
        "train_episode_count": len(train_episodes),
        "val_episode_count": len(val_episodes),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "args": vars(args),
    }
    torch.save(checkpoint_payload(model, metadata), output / "best_model.pt")
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved={output / 'best_model.pt'}", flush=True)


if __name__ == "__main__":
    main()
