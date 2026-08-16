from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train_tfpp_vehicle_steer_adapter import LateralPIDSpec, _derive_yaw_rate, _pid_inputs
from .vehicle_dynamics import InverseDynamicsSpec, SpeedConditionedYawDynamics, bounded_inverse_steer


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline safety diagnostics for the TF++ vehicle dynamics inverse")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--max-delta", type=float, default=0.12)
    parser.add_argument("--base-regularization", type=float, default=0.01)
    parser.add_argument("--yaw-smooth-frames", type=int, default=3)
    args = parser.parse_args()

    cache = np.load(Path(args.cache).expanduser(), allow_pickle=True)
    raw_metadata = cache["metadata"].item()
    metadata = json.loads(raw_metadata) if isinstance(raw_metadata, str) else dict(raw_metadata)
    episode = cache["sample_episode"].astype(np.int64)
    frame = cache["sample_frame"].astype(np.int64)
    checkpoint_flat = cache["checkpoint_flat"].astype(np.float32)
    scalar = cache["scalar"].astype(np.float32)
    pid = _pid_inputs(checkpoint_flat, scalar, episode, frame, LateralPIDSpec())
    yaw_rate = _derive_yaw_rate(metadata, episode, frame, args.episode_root)
    for episode_id in np.unique(episode):
        ids = np.flatnonzero(episode == episode_id)
        ids = ids[np.argsort(frame[ids])]
        raw = yaw_rate[ids].copy()
        width = max(args.yaw_smooth_frames, 1)
        yaw_rate[ids] = np.asarray([raw[max(0, i - width + 1) : i + 1].mean() for i in range(len(raw))])

    payload = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu")
    model = SpeedConditionedYawDynamics()
    model.load_state_dict(payload["vehicle_dynamics_state"])
    model.eval()
    spec = InverseDynamicsSpec(max_delta=args.max_delta, base_regularization=args.base_regularization)
    results = [
        bounded_inverse_steer(model, cp, float(speed), float(rate), float(base), spec)
        for cp, speed, rate, base in zip(checkpoint_flat, scalar[:, 0], yaw_rate, pid["base_steer"])
    ]
    values = {key: np.asarray([row[key] for row in results]) for key in results[0]}
    active = values["gate"] > 0
    full = values["gate"] >= 0.999
    delta = values["delta"]
    base_error = np.abs(values["predicted_base_yaw_rate"] - values["desired_yaw_rate"])
    adapted_error = np.abs(values["predicted_adapted_yaw_rate"] - values["desired_yaw_rate"])
    report = {
        "sample_count": int(len(delta)),
        "active_count": int(active.sum()),
        "active_fraction": float(active.mean()),
        "full_gate_count": int(full.sum()),
        "delta_mean": float(delta.mean()),
        "delta_abs_mean": float(np.abs(delta).mean()),
        "delta_abs_active_mean": float(np.abs(delta[active]).mean()) if np.any(active) else 0.0,
        "delta_abs_p95": float(np.quantile(np.abs(delta), 0.95)),
        "delta_abs_p99": float(np.quantile(np.abs(delta), 0.99)),
        "delta_bound_fraction": float((np.abs(delta) >= args.max_delta * 0.99).mean()),
        "desired_yaw_rate_abs_p95": float(np.quantile(np.abs(values["desired_yaw_rate"]), 0.95)),
        "predicted_base_yaw_mae": float(base_error.mean()),
        "predicted_adapted_yaw_mae": float(adapted_error.mean()),
        "predicted_active_base_yaw_mae": float(base_error[active].mean()) if np.any(active) else 0.0,
        "predicted_active_adapted_yaw_mae": float(adapted_error[active].mean()) if np.any(active) else 0.0,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
