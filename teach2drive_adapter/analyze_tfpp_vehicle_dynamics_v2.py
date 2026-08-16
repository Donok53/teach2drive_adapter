from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train_tfpp_vehicle_steer_adapter import LateralPIDSpec, _derive_yaw_rate, _pid_inputs
from .vehicle_dynamics import (
    BoundedYawRateCalibrator,
    InverseDynamicsSpec,
    InverseDynamicsV2Spec,
    SpeedConditionedYawDynamics,
    bounded_inverse_steer,
    bounded_inverse_steer_v2,
)


def _values(results: list[dict[str, float]]) -> dict[str, np.ndarray]:
    return {key: np.asarray([row[key] for row in results], dtype=np.float64) for key in results[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v1 and safety-gated dynamics v2 offline")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dynamics-checkpoint", required=True)
    parser.add_argument("--calibrator-checkpoint", required=True)
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--yaw-smooth-frames", type=int, default=3)
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
    for episode_id in np.unique(episode):
        ids = np.flatnonzero(episode == episode_id)
        ids = ids[np.argsort(frame[ids])]
        raw = yaw_rate[ids].copy()
        width = max(args.yaw_smooth_frames, 1)
        yaw_rate[ids] = np.asarray([raw[max(0, i - width + 1) : i + 1].mean() for i in range(len(raw))])

    dynamics_payload = torch.load(Path(args.dynamics_checkpoint).expanduser(), map_location="cpu")
    dynamics = SpeedConditionedYawDynamics()
    dynamics.load_state_dict(dynamics_payload["vehicle_dynamics_state"])
    dynamics.eval()
    calibrator_payload = torch.load(Path(args.calibrator_checkpoint).expanduser(), map_location="cpu")
    calibrator_metadata = calibrator_payload.get("metadata", {}).get("yaw_rate_calibrator", {})
    calibrator = BoundedYawRateCalibrator(
        float(calibrator_metadata.get("minimum_gain", 0.75)),
        float(calibrator_metadata.get("maximum_gain", 1.15)),
    )
    calibrator.load_state_dict(calibrator_payload["yaw_rate_calibrator_state"])
    calibrator.eval()
    v1_spec = InverseDynamicsSpec()
    v2_spec = InverseDynamicsV2Spec()
    speed = scalar[:, 0]
    brake = (target_speed < 0.01) | ((speed / np.maximum(target_speed, 1e-3)) > 1.1)
    v1 = _values(
        [
            bounded_inverse_steer(dynamics, cp, float(v), float(rate), float(base), v1_spec)
            for cp, v, rate, base in zip(checkpoints, speed, yaw_rate, pid["base_steer"])
        ]
    )
    v2 = _values(
        [
            bounded_inverse_steer_v2(
                dynamics, calibrator, cp, float(v), float(rate), float(base), float(target), bool(stop), v2_spec
            )
            for cp, v, rate, base, target, stop in zip(
                checkpoints, speed, yaw_rate, pid["base_steer"], target_speed, brake
            )
        ]
    )
    active_v1, active_v2 = v1["gate"] > 0, v2["gate"] > 0
    entry = v2["entry_strength"] >= 0.5
    exit_phase = v2["exit_strength"] >= 0.5
    report = {
        "sample_count": int(len(speed)),
        "v1_active_fraction": float(active_v1.mean()),
        "v2_active_fraction": float(active_v2.mean()),
        "v1_delta_abs_mean": float(np.abs(v1["delta"]).mean()),
        "v2_delta_abs_mean": float(np.abs(v2["delta"]).mean()),
        "v1_delta_abs_p95": float(np.quantile(np.abs(v1["delta"]), 0.95)),
        "v2_delta_abs_p95": float(np.quantile(np.abs(v2["delta"]), 0.95)),
        "v1_bound_fraction": float((np.abs(v1["delta"]) >= 0.1188).mean()),
        "v2_bound_fraction": float((np.abs(v2["delta"]) >= 0.1188).mean()),
        "gain_mean": float(v2["calibration_gain"].mean()),
        "gain_p05": float(np.quantile(v2["calibration_gain"], 0.05)),
        "gain_p95": float(np.quantile(v2["calibration_gain"], 0.95)),
        "risk_gate_off_fraction": float((v2["risk_gate"] == 0).mean()),
        "overshoot_gate_off_fraction": float((v2["overshoot_gate"] == 0).mean()),
        "exit_count": int(exit_phase.sum()),
        "exit_active_fraction": float(active_v2[exit_phase].mean()) if np.any(exit_phase) else 0.0,
        "entry_count": int(entry.sum()),
        "entry_v1_delta_abs_mean": float(np.abs(v1["delta"][entry]).mean()) if np.any(entry) else 0.0,
        "entry_v2_delta_abs_mean": float(np.abs(v2["delta"][entry]).mean()) if np.any(entry) else 0.0,
        "risk_delta_exact_zero": bool(np.all(v2["delta"][v2["risk_gate"] == 0] == 0.0)),
        "overshoot_delta_exact_zero": bool(np.all(v2["delta"][v2["overshoot_gate"] == 0] == 0.0)),
        "maximum_abs_delta": float(np.abs(v2["delta"]).max()),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
