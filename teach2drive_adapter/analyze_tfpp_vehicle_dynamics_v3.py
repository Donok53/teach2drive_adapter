from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train_tfpp_vehicle_steer_adapter import LateralPIDSpec, _derive_yaw_rate, _pid_inputs
from .vehicle_dynamics_v3 import (
    InverseDynamicsV3Spec,
    LaggedSpeedConditionedYawDynamics,
    bounded_horizon_inverse_steer_v3,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline safety analysis for horizon-aligned dynamics v3")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--blend", type=float, default=0.5)
    args = parser.parse_args()

    cache = np.load(Path(args.cache).expanduser(), allow_pickle=True)
    raw_metadata = cache["metadata"].item()
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else dict(raw_metadata)
    episode = cache["sample_episode"].astype(np.int64)
    frame = cache["sample_frame"].astype(np.int64)
    checkpoints = cache["checkpoint_flat"].astype(np.float32)
    scalar = cache["scalar"].astype(np.float32)
    target_speed = cache["expected_speed"].reshape(-1).astype(np.float32)
    pid = _pid_inputs(checkpoints, scalar, episode, frame, LateralPIDSpec())
    yaw_rate = _derive_yaw_rate(metadata, episode, frame, args.episode_root)
    speed = scalar[:, 0]
    base_steer = pid["base_steer"].astype(np.float64)
    lag1, lag2 = base_steer.copy(), base_steer.copy()
    for episode_id in np.unique(episode):
        ids = np.flatnonzero(episode == episode_id)
        ids = ids[np.argsort(frame[ids])]
        raw = yaw_rate[ids].copy()
        yaw_rate[ids] = np.asarray([raw[max(0, index - 2) : index + 1].mean() for index in range(len(raw))])
        lag1[ids[1:]] = base_steer[ids[:-1]]
        lag2[ids[1:]] = lag1[ids[:-1]]

    payload = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu")
    model = LaggedSpeedConditionedYawDynamics()
    model.load_state_dict(payload["lagged_vehicle_dynamics_state"])
    model.eval()
    spec = InverseDynamicsV3Spec()
    brake = (target_speed < 0.01) | ((speed / np.maximum(target_speed, 1e-3)) > 1.1)
    results = [
        bounded_horizon_inverse_steer_v3(
            model,
            checkpoint,
            float(v),
            float(rate),
            float(base),
            float(previous1),
            float(previous2),
            float(target),
            bool(stop),
            spec,
        )
        for checkpoint, v, rate, base, previous1, previous2, target, stop in zip(
            checkpoints, speed, yaw_rate, base_steer, lag1, lag2, target_speed, brake
        )
    ]
    delta = np.asarray([float(row["delta"]) for row in results])
    gate = np.asarray([float(row["gate"]) for row in results])
    risk_gate = np.asarray([float(row["risk_gate"]) for row in results])
    desired = np.asarray([row["desired_yaw_rate"] for row in results])
    predicted_base = np.asarray([row["predicted_base_yaw_rate"] for row in results])
    predicted_adapted = np.asarray([row["predicted_adapted_yaw_rate"] for row in results])
    weights = np.asarray(spec.horizon_weights, dtype=np.float64)
    base_error = np.sqrt(np.average((predicted_base - desired) ** 2, axis=1, weights=weights))
    adapted_error = np.sqrt(np.average((predicted_adapted - desired) ** 2, axis=1, weights=weights))
    applied = float(args.blend) * delta
    active = gate > 1e-8
    report = {
        "sample_count": int(len(results)),
        "active_fraction": float(active.mean()),
        "risk_reduced_fraction": float((risk_gate < 1.0).mean()),
        "preblend_abs_delta_mean": float(np.abs(delta).mean()),
        "preblend_abs_delta_p95": float(np.quantile(np.abs(delta), 0.95)),
        "preblend_bound_fraction": float((np.abs(delta) >= 0.1188).mean()),
        "applied_abs_delta_mean": float(np.abs(applied).mean()),
        "applied_abs_delta_p95": float(np.quantile(np.abs(applied), 0.95)),
        "maximum_abs_applied_delta": float(np.abs(applied).max()),
        "base_tracking_rmse_mean": float(base_error.mean()),
        "adapted_tracking_rmse_mean": float(adapted_error.mean()),
        "tracking_improvement_fraction": float(1.0 - adapted_error.mean() / max(base_error.mean(), 1e-12)),
        "worsened_tracking_fraction": float((adapted_error > base_error + 1e-9).mean()),
        "desired_yaw_rate_abs_p95": float(np.quantile(np.abs(desired), 0.95)),
    }
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output, flush=True)
    if args.out:
        Path(args.out).expanduser().write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
