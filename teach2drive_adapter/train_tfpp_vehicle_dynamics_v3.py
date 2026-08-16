from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .vehicle_dynamics import SpeedConditionedYawDynamics
from .vehicle_dynamics_v3 import LaggedSpeedConditionedYawDynamics, checkpoint_payload


def _load_episode(path: Path, smooth_frames: int) -> dict[str, np.ndarray] | None:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    if len(rows) < 8:
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
    yaw_rate = np.asarray([raw_rate[max(0, index - width + 1) : index + 1].mean() for index in range(len(raw_rate))])
    return {"speed": speed, "steer": steer, "brake": brake, "yaw_rate": yaw_rate, "dt": dt}


def _features(
    speed: np.ndarray,
    yaw_rate: np.ndarray,
    steer: np.ndarray,
    steer_lag1: np.ndarray,
    steer_lag2: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        (
            yaw_rate,
            (speed / 20.0) * yaw_rate,
            np.ones_like(speed),
            speed * steer,
            (speed**2 / 20.0) * steer,
            steer,
            steer**3,
            speed * steer_lag1,
            steer_lag1,
            speed * steer_lag2,
            steer_lag2,
        )
    )


def _pack(episodes: list[dict[str, np.ndarray]], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, targets, steering = [], [], []
    for episode in episodes:
        speed, steer, brake, yaw_rate = (episode[key] for key in ("speed", "steer", "brake", "yaw_rate"))
        current = slice(2, -1)
        mask = (
            (speed[current] >= args.minimum_speed)
            & (brake[current] < args.maximum_brake)
            & (np.abs(steer[current]) <= args.maximum_abs_steer)
            & (np.abs(yaw_rate[current]) <= args.maximum_abs_yaw_rate)
            & (np.abs(yaw_rate[3:]) <= args.maximum_abs_yaw_rate)
        )
        features.append(
            _features(
                speed[current][mask],
                yaw_rate[current][mask],
                steer[current][mask],
                steer[1:-2][mask],
                steer[:-3][mask],
            )
        )
        targets.append(yaw_rate[3:][mask])
        steering.append(np.maximum.reduce((np.abs(steer[current][mask]), np.abs(steer[1:-2][mask]), np.abs(steer[:-3][mask]))))
    return np.concatenate(features), np.concatenate(targets), np.concatenate(steering)


def _robust_weighted_ridge(x: np.ndarray, y: np.ndarray, steer: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    turn = np.clip(np.abs(steer) / max(args.turn_scale, 1e-6), 0.0, 1.0)
    base_weight = 1.0 + args.turn_weight * turn
    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    eye = np.eye(x.shape[1], dtype=np.float64)
    eye[2, 2] = 0.0
    robust = np.ones_like(y)
    for _ in range(max(args.irls_iterations, 1)):
        root = np.sqrt(base_weight * robust)
        xw, yw = x * root[:, None], y * root
        coefficients = np.linalg.solve(xw.T @ xw + args.ridge * eye, xw.T @ yw)
        residual = y - x @ coefficients
        scale = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-4)
        robust = np.minimum(1.0, args.huber_delta * scale / np.maximum(np.abs(residual), 1e-8))
    return coefficients


def _metrics(x: np.ndarray, y: np.ndarray, steer: np.ndarray, coefficients: np.ndarray) -> dict[str, float]:
    prediction = x @ coefficients
    turn = np.abs(steer) >= 0.05
    return {
        "count": int(len(y)),
        "mae": float(np.mean(np.abs(y - prediction))),
        "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "r2": float(1.0 - np.sum((y - prediction) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)),
        "persistence_mae": float(np.mean(np.abs(y - x[:, 0]))),
        "turn_mae": float(np.mean(np.abs(y[turn] - prediction[turn]))) if np.any(turn) else math.nan,
        "turn_count": int(turn.sum()),
    }


def _rollout_metrics(
    episodes: list[dict[str, np.ndarray]],
    model: LaggedSpeedConditionedYawDynamics,
    baseline: SpeedConditionedYawDynamics | None,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for horizon in range(1, args.rollout_steps + 1):
        v3_error, v1_error, turn_mask = [], [], []
        for episode in episodes:
            speed, steer, brake, yaw_rate = (episode[key] for key in ("speed", "steer", "brake", "yaw_rate"))
            for index in range(2, len(yaw_rate) - horizon):
                if speed[index] < args.minimum_speed or brake[index] >= args.maximum_brake:
                    continue
                if np.max(np.abs(yaw_rate[index : index + horizon + 1])) > args.maximum_abs_yaw_rate:
                    continue
                predicted_v3 = float(yaw_rate[index])
                predicted_v1 = float(yaw_rate[index])
                for step in range(horizon):
                    at = index + step
                    predicted_v3 = float(
                        model.predict_numpy(speed[at], predicted_v3, steer[at], steer[at - 1], steer[at - 2])
                    )
                    if baseline is not None:
                        predicted_v1 = float(baseline.predict_numpy(speed[at], predicted_v1, steer[at]))
                target = float(yaw_rate[index + horizon])
                v3_error.append(abs(target - predicted_v3))
                if baseline is not None:
                    v1_error.append(abs(target - predicted_v1))
                turn_mask.append(abs(float(steer[index])) >= 0.05)
        turn = np.asarray(turn_mask, dtype=bool)
        v3 = np.asarray(v3_error, dtype=np.float64)
        values = {
            "count": int(len(v3)),
            "v3_mae": float(v3.mean()),
            "v3_turn_mae": float(v3[turn].mean()) if np.any(turn) else math.nan,
        }
        if v1_error:
            v1 = np.asarray(v1_error, dtype=np.float64)
            values.update(
                {
                    "v1_mae": float(v1.mean()),
                    "v1_turn_mae": float(v1[turn].mean()) if np.any(turn) else math.nan,
                    "mae_reduction_fraction": float(1.0 - v3.mean() / max(v1.mean(), 1e-12)),
                }
            )
        report[f"horizon_{horizon}"] = values
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit steering-memory Tesla yaw dynamics for TF++ v3")
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--v1-checkpoint", default="")
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
    parser.add_argument("--rollout-steps", type=int, default=3)
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
    train = [data for index, (_, data) in enumerate(loaded) if index not in val_ids]
    validation = [data for index, (_, data) in enumerate(loaded) if index in val_ids]
    x_train, y_train, s_train = _pack(train, args)
    x_val, y_val, s_val = _pack(validation, args)
    coefficients = _robust_weighted_ridge(x_train, y_train, s_train, args)
    model = LaggedSpeedConditionedYawDynamics(coefficients)

    baseline = None
    if args.v1_checkpoint:
        payload = torch.load(Path(args.v1_checkpoint).expanduser(), map_location="cpu")
        baseline = SpeedConditionedYawDynamics()
        baseline.load_state_dict(payload["vehicle_dynamics_state"])
        baseline.eval()
    train_metrics = _metrics(x_train, y_train, s_train, coefficients)
    val_metrics = _metrics(x_val, y_val, s_val, coefficients)
    rollout_metrics = _rollout_metrics(validation, model, baseline, args)
    metadata = {
        "feature_names": list(model.feature_names),
        "coefficients": coefficients.tolist(),
        "step_seconds": float(np.median(np.concatenate([data["dt"] for _, data in loaded]))),
        "lag_seconds": [0.25, 0.50],
        "episode_count": len(loaded),
        "train_episode_count": len(train),
        "val_episode_count": len(validation),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "rollout_metrics": rollout_metrics,
        "args": vars(args),
    }
    output = Path(args.out_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, metadata), output / "best_model.pt")
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print(f"saved={output / 'best_model.pt'}", flush=True)


if __name__ == "__main__":
    main()
