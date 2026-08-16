from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .vehicle_dynamics_v3 import LaggedSpeedConditionedYawDynamics, _sample_polyline, rollout_constant_steer


class HorizonYawRateCalibrator(nn.Module):
    """Small bounded gain model for each future yaw-rate horizon."""

    feature_names = ("bias", "speed", "abs_curvature", "entry", "exit")

    def __init__(
        self,
        horizons: int = 3,
        minimum_gain: float = 0.85,
        maximum_gain: float = 1.10,
    ) -> None:
        super().__init__()
        if not minimum_gain < 1.0 < maximum_gain:
            raise ValueError("minimum_gain < 1 < maximum_gain is required")
        self.horizons = int(horizons)
        self.minimum_gain = float(minimum_gain)
        self.maximum_gain = float(maximum_gain)
        identity_fraction = (1.0 - minimum_gain) / (maximum_gain - minimum_gain)
        identity_logit = math.log(identity_fraction / (1.0 - identity_fraction))
        initial = torch.zeros((self.horizons, len(self.feature_names)), dtype=torch.float32)
        initial[:, 0] = identity_logit
        self.coefficients = nn.Parameter(initial)

    @staticmethod
    def features(
        speed: torch.Tensor,
        curvature: torch.Tensor,
        entry: torch.Tensor,
        exit_phase: torch.Tensor,
    ) -> torch.Tensor:
        speed = speed.float().reshape(-1, 1)
        curvature = curvature.float()
        if curvature.ndim == 1:
            curvature = curvature.reshape(-1, 1)
        entry = entry.float().reshape(-1, 1).expand_as(curvature)
        exit_phase = exit_phase.float().reshape(-1, 1).expand_as(curvature)
        return torch.stack(
            (
                torch.ones_like(curvature),
                (speed / 15.0).clamp(0.0, 2.0).expand_as(curvature),
                (curvature.abs() / 0.10).clamp(0.0, 3.0),
                entry.clamp(0.0, 1.0),
                exit_phase.clamp(0.0, 1.0),
            ),
            dim=-1,
        )

    def forward(
        self,
        speed: torch.Tensor,
        curvature: torch.Tensor,
        entry: torch.Tensor,
        exit_phase: torch.Tensor,
    ) -> torch.Tensor:
        features = self.features(speed, curvature, entry, exit_phase)
        if features.shape[1] != self.horizons:
            raise ValueError(f"Expected {self.horizons} horizons, got {features.shape[1]}")
        raw = (features * self.coefficients.unsqueeze(0)).sum(dim=-1)
        return self.minimum_gain + (self.maximum_gain - self.minimum_gain) * torch.sigmoid(raw)

    def predict_numpy(
        self,
        speed: float,
        curvature: np.ndarray,
        entry: float,
        exit_phase: float,
    ) -> np.ndarray:
        curvature = np.asarray(curvature, dtype=np.float64).reshape(1, -1)
        with torch.no_grad():
            result = self(
                torch.tensor([speed], dtype=torch.float32),
                torch.as_tensor(curvature, dtype=torch.float32),
                torch.tensor([entry], dtype=torch.float32),
                torch.tensor([exit_phase], dtype=torch.float32),
            )
        return result.cpu().numpy().reshape(-1).astype(np.float64)


@dataclass(frozen=True)
class InverseDynamicsV4Spec:
    horizons_seconds: tuple[float, ...] = (0.25, 0.50, 0.75)
    horizon_weights: tuple[float, ...] = (1.0, 1.5, 2.0)
    max_delta: float = 0.12
    minimum_speed: float = 1.0
    minimum_target_speed: float = 2.0
    turn_threshold_yaw_rate: float = 0.03
    full_turn_threshold_yaw_rate: float = 0.12
    maximum_yaw_rate: float = 0.80
    phase_curvature_scale: float = 0.02
    exit_gate_floor: float = 0.0
    overshoot_ratio: float = 0.95
    overshoot_minimum_yaw_rate: float = 0.03
    grid_size: int = 257
    base_regularization: float = 0.01


def checkpoint_pure_pursuit_sequence(
    checkpoints: np.ndarray,
    speed: float,
    horizons_seconds: tuple[float, ...] = (0.25, 0.50, 0.75),
    maximum_yaw_rate: float = 0.80,
    phase_curvature_scale: float = 0.02,
) -> dict[str, Any]:
    """Create horizon-aligned pure-pursuit targets from frozen TF++ checkpoints."""
    raw = np.asarray(checkpoints, dtype=np.float64).reshape(-1, 2)
    if len(raw) == 0:
        zeros = [0.0] * len(horizons_seconds)
        return {"curvature": zeros, "desired_yaw_rate": zeros, "sample_distance": zeros, "entry": 0.0, "exit": 0.0}
    points = np.vstack((np.zeros((1, 2), dtype=np.float64), raw))
    keep = np.concatenate(([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5))
    points = points[keep]
    if len(points) < 2:
        zeros = [0.0] * len(horizons_seconds)
        return {"curvature": zeros, "desired_yaw_rate": zeros, "sample_distance": zeros, "entry": 0.0, "exit": 0.0}
    total = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    curvatures: list[float] = []
    distances: list[float] = []
    for horizon in horizons_seconds:
        requested = max(float(speed), 0.0) * float(horizon)
        distance = float(np.clip(requested, 0.25, total))
        point = _sample_polyline(points, distance)
        point_distance2 = max(float(point @ point), 1e-6)
        curvatures.append(2.0 * float(point[1]) / point_distance2)
        distances.append(distance)
    scale = max(float(phase_curvature_scale), 1e-6)
    phase_signal = (abs(curvatures[-1]) - abs(curvatures[0])) / scale
    entry = float(np.clip(phase_signal, 0.0, 1.0))
    exit_phase = float(np.clip(-phase_signal, 0.0, 1.0))
    desired = np.clip(max(float(speed), 0.0) * np.asarray(curvatures), -maximum_yaw_rate, maximum_yaw_rate)
    return {
        "curvature": [float(value) for value in curvatures],
        "desired_yaw_rate": [float(value) for value in desired],
        "sample_distance": distances,
        "entry": entry,
        "exit": exit_phase,
    }


def bounded_horizon_inverse_steer_v4(
    dynamics: LaggedSpeedConditionedYawDynamics,
    calibrator: HorizonYawRateCalibrator,
    checkpoints: np.ndarray,
    speed: float,
    yaw_rate: float,
    base_steer: float,
    steer_lag1: float,
    steer_lag2: float,
    target_speed: float,
    brake: bool,
    spec: InverseDynamicsV4Spec,
) -> dict[str, Any]:
    target = checkpoint_pure_pursuit_sequence(
        checkpoints,
        speed,
        spec.horizons_seconds,
        spec.maximum_yaw_rate,
        spec.phase_curvature_scale,
    )
    curvature = np.asarray(target["curvature"], dtype=np.float64)
    gains = calibrator.predict_numpy(speed, curvature, target["entry"], target["exit"])
    desired = np.clip(max(float(speed), 0.0) * curvature * gains, -spec.maximum_yaw_rate, spec.maximum_yaw_rate)

    base = float(np.clip(base_steer, -1.0, 1.0))
    lower, upper = max(-1.0, base - spec.max_delta), min(1.0, base + spec.max_delta)
    candidates = np.linspace(lower, upper, max(int(spec.grid_size), 3), dtype=np.float64)
    predicted = rollout_constant_steer(dynamics, speed, yaw_rate, candidates, steer_lag1, steer_lag2, len(desired))
    weights = np.asarray(spec.horizon_weights, dtype=np.float64)
    if len(weights) != len(desired):
        raise ValueError("horizon_weights must match horizons_seconds")
    tracking = ((predicted - desired[None, :]) ** 2 * weights[None, :]).sum(axis=1) / max(weights.sum(), 1e-8)
    scale = max(spec.max_delta, 1e-6)
    objective = tracking + spec.base_regularization * ((candidates - base) / scale) ** 2
    raw = float(candidates[int(np.argmin(objective))])

    turn_signal = max(float(np.max(np.abs(desired), initial=0.0)), abs(float(yaw_rate)))
    width = max(spec.full_turn_threshold_yaw_rate - spec.turn_threshold_yaw_rate, 1e-6)
    turn_gate = float(np.clip((turn_signal - spec.turn_threshold_yaw_rate) / width, 0.0, 1.0))
    if float(speed) < spec.minimum_speed:
        turn_gate = 0.0
    phase_gate = float(1.0 - (1.0 - spec.exit_gate_floor) * float(target["exit"]))
    risk_gate = float(not bool(brake) and float(target_speed) >= spec.minimum_target_speed)
    reference = float(desired[int(np.argmax(np.abs(desired)))]) if len(desired) else 0.0
    same_direction = reference * float(yaw_rate) > 0.0
    overshoot = (
        abs(reference) >= spec.overshoot_minimum_yaw_rate
        and same_direction
        and abs(float(yaw_rate)) >= spec.overshoot_ratio * abs(reference)
    )
    overshoot_gate = 0.0 if overshoot else 1.0
    gate = turn_gate * phase_gate * risk_gate * overshoot_gate
    steer = float(np.clip(base + gate * (raw - base), -1.0, 1.0))
    adapted_prediction = rollout_constant_steer(
        dynamics, speed, yaw_rate, steer, steer_lag1, steer_lag2, len(desired)
    ).reshape(-1)
    base_prediction = rollout_constant_steer(
        dynamics, speed, yaw_rate, base, steer_lag1, steer_lag2, len(desired)
    ).reshape(-1)
    return {
        "steer": steer,
        "delta": steer - base,
        "raw_steer": raw,
        "gate": gate,
        "turn_gate": turn_gate,
        "phase_gate": phase_gate,
        "risk_gate": risk_gate,
        "overshoot_gate": overshoot_gate,
        "calibration_gain": gains.tolist(),
        "desired_yaw_rate": desired.tolist(),
        "raw_desired_yaw_rate": target["desired_yaw_rate"],
        "curvature": target["curvature"],
        "sample_distance": target["sample_distance"],
        "entry_strength": target["entry"],
        "exit_strength": target["exit"],
        "predicted_base_yaw_rate": base_prediction.tolist(),
        "predicted_adapted_yaw_rate": adapted_prediction.tolist(),
        "yaw_rate": float(yaw_rate),
        "speed": float(speed),
        "target_speed": float(target_speed),
        "brake": float(bool(brake)),
        "base_steer": base,
        "steer_lag1": float(steer_lag1),
        "steer_lag2": float(steer_lag2),
    }


def calibrator_checkpoint_payload(model: HorizonYawRateCalibrator, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "horizon_yaw_calibrator_state": model.state_dict(),
        "metadata": {"horizon_yaw_calibrator": metadata},
    }
