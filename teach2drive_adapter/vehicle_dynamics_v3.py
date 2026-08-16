from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


class LaggedSpeedConditionedYawDynamics(nn.Module):
    """Tesla yaw dynamics with 0.25 s and 0.50 s steering-memory terms."""

    feature_names = (
        "yaw_rate",
        "speed_yaw_rate",
        "bias",
        "speed_steer",
        "speed2_steer",
        "steer",
        "steer_cubic",
        "speed_steer_lag1",
        "steer_lag1",
        "speed_steer_lag2",
        "steer_lag2",
    )

    def __init__(self, coefficients: torch.Tensor | np.ndarray | None = None) -> None:
        super().__init__()
        if coefficients is None:
            coefficients = torch.zeros(len(self.feature_names), dtype=torch.float32)
        coefficients = torch.as_tensor(coefficients, dtype=torch.float32).reshape(-1)
        if coefficients.numel() != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} coefficients, got {coefficients.numel()}")
        self.coefficients = nn.Parameter(coefficients.clone())

    @staticmethod
    def features(
        speed: torch.Tensor,
        yaw_rate: torch.Tensor,
        steer: torch.Tensor,
        steer_lag1: torch.Tensor,
        steer_lag2: torch.Tensor,
    ) -> torch.Tensor:
        speed = speed.float().reshape(-1)
        yaw_rate = yaw_rate.float().reshape(-1)
        steer = steer.float().reshape(-1)
        steer_lag1 = steer_lag1.float().reshape(-1)
        steer_lag2 = steer_lag2.float().reshape(-1)
        return torch.stack(
            (
                yaw_rate,
                (speed / 20.0) * yaw_rate,
                torch.ones_like(speed),
                speed * steer,
                (speed.square() / 20.0) * steer,
                steer,
                steer.pow(3),
                speed * steer_lag1,
                steer_lag1,
                speed * steer_lag2,
                steer_lag2,
            ),
            dim=-1,
        )

    def forward(
        self,
        speed: torch.Tensor,
        yaw_rate: torch.Tensor,
        steer: torch.Tensor,
        steer_lag1: torch.Tensor,
        steer_lag2: torch.Tensor,
    ) -> torch.Tensor:
        return self.features(speed, yaw_rate, steer, steer_lag1, steer_lag2) @ self.coefficients

    def predict_numpy(
        self,
        speed: float,
        yaw_rate: np.ndarray | float,
        steer: np.ndarray | float,
        steer_lag1: np.ndarray | float,
        steer_lag2: np.ndarray | float,
    ) -> np.ndarray:
        c = self.coefficients.detach().cpu().numpy().astype(np.float64)
        v = float(speed)
        r, s, s1, s2 = np.broadcast_arrays(
            np.asarray(yaw_rate, dtype=np.float64),
            np.asarray(steer, dtype=np.float64),
            np.asarray(steer_lag1, dtype=np.float64),
            np.asarray(steer_lag2, dtype=np.float64),
        )
        return (
            c[0] * r
            + c[1] * (v / 20.0) * r
            + c[2]
            + c[3] * v * s
            + c[4] * (v * v / 20.0) * s
            + c[5] * s
            + c[6] * s**3
            + c[7] * v * s1
            + c[8] * s1
            + c[9] * v * s2
            + c[10] * s2
        )


@dataclass(frozen=True)
class InverseDynamicsV3Spec:
    horizons_seconds: tuple[float, ...] = (0.25, 0.50, 0.75)
    horizon_weights: tuple[float, ...] = (1.0, 1.5, 2.0)
    max_delta: float = 0.12
    minimum_speed: float = 1.0
    turn_threshold_yaw_rate: float = 0.03
    full_turn_threshold_yaw_rate: float = 0.12
    maximum_yaw_rate: float = 0.80
    curvature_probe_distance: float = 0.75
    risk_gate_floor: float = 0.25
    minimum_target_speed: float = 1.0
    grid_size: int = 257
    base_regularization: float = 0.01


def _sample_polyline(points: np.ndarray, distance: float) -> np.ndarray:
    delta = np.diff(points, axis=0)
    segment = np.linalg.norm(delta, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    if cumulative[-1] <= 1e-8:
        return points[0].copy()
    target = float(np.clip(distance, 0.0, cumulative[-1]))
    index = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(segment) - 1)
    width = max(float(segment[index]), 1e-8)
    alpha = (target - float(cumulative[index])) / width
    return points[index] + alpha * delta[index]


def checkpoint_curvature_sequence(
    checkpoints: np.ndarray,
    speed: float,
    horizons_seconds: tuple[float, ...] = (0.25, 0.50, 0.75),
    probe_distance: float = 0.75,
    maximum_yaw_rate: float = 0.80,
) -> dict[str, Any]:
    """Return path-local curvature/yaw targets at matching future distances."""
    raw = np.asarray(checkpoints, dtype=np.float64).reshape(-1, 2)
    if len(raw) == 0:
        zeros = [0.0] * len(horizons_seconds)
        return {"curvature": zeros, "desired_yaw_rate": zeros, "sample_distance": zeros}
    points = np.vstack((np.zeros((1, 2), dtype=np.float64), raw))
    keep = np.concatenate(([True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-5))
    points = points[keep]
    if len(points) < 3:
        zeros = [0.0] * len(horizons_seconds)
        return {"curvature": zeros, "desired_yaw_rate": zeros, "sample_distance": zeros}
    total = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    probe = max(float(probe_distance), 0.10)
    curvatures: list[float] = []
    distances: list[float] = []
    for horizon in horizons_seconds:
        center_distance = float(np.clip(max(float(speed), 0.0) * float(horizon), 0.10, total))
        before = _sample_polyline(points, max(0.0, center_distance - probe))
        center = _sample_polyline(points, center_distance)
        after = _sample_polyline(points, min(total, center_distance + probe))
        ab, ac, bc = center - before, after - before, after - center
        denominator = float(np.linalg.norm(ab) * np.linalg.norm(ac) * np.linalg.norm(bc))
        curvature = 0.0 if denominator < 1e-8 else 2.0 * float(np.cross(ab, ac)) / denominator
        curvatures.append(curvature)
        distances.append(center_distance)
    desired = np.clip(max(float(speed), 0.0) * np.asarray(curvatures), -maximum_yaw_rate, maximum_yaw_rate)
    return {
        "curvature": [float(value) for value in curvatures],
        "desired_yaw_rate": [float(value) for value in desired],
        "sample_distance": distances,
    }


def rollout_constant_steer(
    model: LaggedSpeedConditionedYawDynamics,
    speed: float,
    yaw_rate: float,
    steer: np.ndarray | float,
    steer_lag1: float,
    steer_lag2: float,
    steps: int,
) -> np.ndarray:
    steer_array = np.asarray(steer, dtype=np.float64)
    current = np.broadcast_to(np.asarray(yaw_rate, dtype=np.float64), steer_array.shape).copy()
    lag1 = np.broadcast_to(np.asarray(steer_lag1, dtype=np.float64), steer_array.shape).copy()
    lag2 = np.broadcast_to(np.asarray(steer_lag2, dtype=np.float64), steer_array.shape).copy()
    predictions = []
    for _ in range(max(int(steps), 1)):
        current = model.predict_numpy(speed, current, steer_array, lag1, lag2)
        predictions.append(current.copy())
        lag2, lag1 = lag1, steer_array
    return np.stack(predictions, axis=-1)


def bounded_horizon_inverse_steer_v3(
    model: LaggedSpeedConditionedYawDynamics,
    checkpoints: np.ndarray,
    speed: float,
    yaw_rate: float,
    base_steer: float,
    steer_lag1: float,
    steer_lag2: float,
    target_speed: float,
    brake: bool,
    spec: InverseDynamicsV3Spec,
) -> dict[str, Any]:
    target = checkpoint_curvature_sequence(
        checkpoints,
        speed,
        spec.horizons_seconds,
        spec.curvature_probe_distance,
        spec.maximum_yaw_rate,
    )
    desired = np.asarray(target["desired_yaw_rate"], dtype=np.float64)
    base = float(np.clip(base_steer, -1.0, 1.0))
    lower, upper = max(-1.0, base - spec.max_delta), min(1.0, base + spec.max_delta)
    candidates = np.linspace(lower, upper, max(int(spec.grid_size), 3), dtype=np.float64)
    predicted = rollout_constant_steer(
        model, speed, yaw_rate, candidates, steer_lag1, steer_lag2, len(spec.horizons_seconds)
    )
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
    risk = bool(brake) or float(target_speed) < spec.minimum_target_speed
    risk_gate = float(spec.risk_gate_floor if risk else 1.0)
    gate = turn_gate * risk_gate
    steer = float(np.clip(base + gate * (raw - base), -1.0, 1.0))
    adapted_prediction = rollout_constant_steer(
        model, speed, yaw_rate, steer, steer_lag1, steer_lag2, len(spec.horizons_seconds)
    ).reshape(-1)
    base_prediction = rollout_constant_steer(
        model, speed, yaw_rate, base, steer_lag1, steer_lag2, len(spec.horizons_seconds)
    ).reshape(-1)
    return {
        "steer": steer,
        "delta": steer - base,
        "raw_steer": raw,
        "gate": gate,
        "turn_gate": turn_gate,
        "risk_gate": risk_gate,
        "desired_yaw_rate": desired.tolist(),
        "curvature": target["curvature"],
        "sample_distance": target["sample_distance"],
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


def checkpoint_payload(model: LaggedSpeedConditionedYawDynamics, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "lagged_vehicle_dynamics_state": model.state_dict(),
        "metadata": {"lagged_vehicle_dynamics": metadata},
    }
