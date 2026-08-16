from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from torch import nn


class SpeedConditionedYawDynamics(nn.Module):
    """One-step steer-to-yaw-rate model with an interpretable polynomial basis.

    The seven features are deliberately small and planner independent.  They
    describe yaw inertia plus a speed-conditioned, odd steering response:

      r[t+1] = c0*r + c1*(v/20)*r + c2
               + c3*v*s + c4*(v**2/20)*s + c5*s + c6*s**3

    where ``v`` is m/s, ``s`` is normalized steering and ``r`` is rad/s.
    """

    feature_names = (
        "yaw_rate",
        "speed_yaw_rate",
        "bias",
        "speed_steer",
        "speed2_steer",
        "steer",
        "steer_cubic",
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
    def features(speed: torch.Tensor, yaw_rate: torch.Tensor, steer: torch.Tensor) -> torch.Tensor:
        speed = speed.float().reshape(-1)
        yaw_rate = yaw_rate.float().reshape(-1)
        steer = steer.float().reshape(-1)
        return torch.stack(
            (
                yaw_rate,
                (speed / 20.0) * yaw_rate,
                torch.ones_like(speed),
                speed * steer,
                (speed.square() / 20.0) * steer,
                steer,
                steer.pow(3),
            ),
            dim=-1,
        )

    def forward(self, speed: torch.Tensor, yaw_rate: torch.Tensor, steer: torch.Tensor) -> torch.Tensor:
        return self.features(speed, yaw_rate, steer) @ self.coefficients

    def predict_numpy(self, speed: float, yaw_rate: float, steer: np.ndarray | float) -> np.ndarray:
        c = self.coefficients.detach().cpu().numpy().astype(np.float64)
        s = np.asarray(steer, dtype=np.float64)
        v = float(speed)
        r = float(yaw_rate)
        state = c[0] * r + c[1] * (v / 20.0) * r + c[2]
        linear = c[3] * v + c[4] * (v * v / 20.0) + c[5]
        return state + linear * s + c[6] * s**3


class BoundedYawRateCalibrator(nn.Module):
    """Small, bounded calibration of checkpoint-derived desired yaw rate.

    This module never sees expert steering.  It only learns how the geometric
    curvature of frozen TF++ checkpoints should be scaled to match the future
    spatial path recorded from the target vehicle.
    """

    feature_names = ("bias", "speed", "abs_curvature", "entry", "exit")

    def __init__(self, minimum_gain: float = 0.75, maximum_gain: float = 1.15) -> None:
        super().__init__()
        if not minimum_gain < 1.0 < maximum_gain:
            raise ValueError("minimum_gain < 1 < maximum_gain is required")
        self.minimum_gain = float(minimum_gain)
        self.maximum_gain = float(maximum_gain)
        identity_fraction = (1.0 - self.minimum_gain) / (self.maximum_gain - self.minimum_gain)
        identity_logit = math.log(identity_fraction / (1.0 - identity_fraction))
        initial = torch.zeros(len(self.feature_names), dtype=torch.float32)
        initial[0] = identity_logit
        self.coefficients = nn.Parameter(initial)

    @staticmethod
    def features(
        speed: torch.Tensor,
        curvature: torch.Tensor,
        entry_strength: torch.Tensor,
        exit_strength: torch.Tensor,
    ) -> torch.Tensor:
        speed = speed.float().reshape(-1)
        curvature = curvature.float().reshape(-1)
        entry_strength = entry_strength.float().reshape(-1)
        exit_strength = exit_strength.float().reshape(-1)
        return torch.stack(
            (
                torch.ones_like(speed),
                (speed / 15.0).clamp(0.0, 2.0),
                (curvature.abs() / 0.10).clamp(0.0, 3.0),
                entry_strength.clamp(0.0, 1.0),
                exit_strength.clamp(0.0, 1.0),
            ),
            dim=-1,
        )

    def forward(
        self,
        speed: torch.Tensor,
        curvature: torch.Tensor,
        entry_strength: torch.Tensor,
        exit_strength: torch.Tensor,
    ) -> torch.Tensor:
        raw = self.features(speed, curvature, entry_strength, exit_strength) @ self.coefficients
        return self.minimum_gain + (self.maximum_gain - self.minimum_gain) * torch.sigmoid(raw)

    def predict_numpy(self, speed: float, curvature: float, entry_strength: float, exit_strength: float) -> float:
        features = np.asarray(
            [
                1.0,
                np.clip(float(speed) / 15.0, 0.0, 2.0),
                np.clip(abs(float(curvature)) / 0.10, 0.0, 3.0),
                np.clip(float(entry_strength), 0.0, 1.0),
                np.clip(float(exit_strength), 0.0, 1.0),
            ],
            dtype=np.float64,
        )
        coefficients = self.coefficients.detach().cpu().numpy().astype(np.float64)
        raw = float(features @ coefficients)
        sigmoid = 1.0 / (1.0 + math.exp(-float(np.clip(raw, -30.0, 30.0))))
        return self.minimum_gain + (self.maximum_gain - self.minimum_gain) * sigmoid


@dataclass(frozen=True)
class InverseDynamicsSpec:
    max_delta: float = 0.12
    minimum_speed: float = 1.0
    turn_threshold_rad: float = 0.04
    full_turn_threshold_rad: float = 0.12
    maximum_yaw_rate: float = 0.80
    grid_size: int = 257
    base_regularization: float = 0.01


@dataclass(frozen=True)
class InverseDynamicsV2Spec(InverseDynamicsSpec):
    phase_curvature_scale: float = 0.02
    exit_gate_floor: float = 0.0
    minimum_target_speed: float = 2.0
    overshoot_ratio: float = 0.95
    overshoot_minimum_yaw_rate: float = 0.03


def checkpoint_desired_yaw_rate(checkpoints: np.ndarray, speed: float, maximum_yaw_rate: float = 0.8) -> tuple[float, float]:
    """Return pure-pursuit yaw-rate and heading error for TF++ checkpoints."""
    points = np.asarray(checkpoints, dtype=np.float64).reshape(-1, 2)
    if points.size == 0:
        return 0.0, 0.0
    point = points[-1]
    distance = max(float(np.linalg.norm(point)), 1e-3)
    heading = math.atan2(float(point[1]), float(point[0]))
    desired = 2.0 * max(float(speed), 0.0) * math.sin(heading) / distance
    return float(np.clip(desired, -maximum_yaw_rate, maximum_yaw_rate)), float(heading)


def checkpoint_curvature_phase(checkpoints: np.ndarray, phase_curvature_scale: float = 0.02) -> Dict[str, float]:
    """Infer turn entry/exit from the near-to-far checkpoint curvature profile."""
    points = np.asarray(checkpoints, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return {"curvature": 0.0, "near_curvature": 0.0, "heading_error": 0.0, "entry": 0.0, "exit": 0.0}

    def curvature(point: np.ndarray) -> float:
        distance2 = max(float(point @ point), 1e-6)
        return 2.0 * float(point[1]) / distance2

    far = points[-1]
    near = points[min(4, len(points) - 1)]
    far_curvature = curvature(far)
    near_curvature = curvature(near)
    scale = max(float(phase_curvature_scale), 1e-6)
    phase_signal = (abs(far_curvature) - abs(near_curvature)) / scale
    return {
        "curvature": far_curvature,
        "near_curvature": near_curvature,
        "heading_error": math.atan2(float(far[1]), float(far[0])),
        "entry": float(np.clip(phase_signal, 0.0, 1.0)),
        "exit": float(np.clip(-phase_signal, 0.0, 1.0)),
    }


def bounded_inverse_steer(
    model: SpeedConditionedYawDynamics,
    checkpoints: np.ndarray,
    speed: float,
    yaw_rate: float,
    base_steer: float,
    spec: InverseDynamicsSpec,
) -> Dict[str, float]:
    """Solve a safe one-dimensional inverse problem around the stock PID steer."""
    desired, heading = checkpoint_desired_yaw_rate(checkpoints, speed, spec.maximum_yaw_rate)
    strength = abs(heading)
    width = max(spec.full_turn_threshold_rad - spec.turn_threshold_rad, 1e-6)
    gate = float(np.clip((strength - spec.turn_threshold_rad) / width, 0.0, 1.0))
    if float(speed) < spec.minimum_speed:
        gate = 0.0

    base = float(np.clip(base_steer, -1.0, 1.0))
    lower = max(-1.0, base - spec.max_delta)
    upper = min(1.0, base + spec.max_delta)
    candidates = np.linspace(lower, upper, max(int(spec.grid_size), 3), dtype=np.float64)
    predicted = model.predict_numpy(speed, yaw_rate, candidates)
    scale = max(spec.max_delta, 1e-6)
    objective = (predicted - desired) ** 2 + spec.base_regularization * ((candidates - base) / scale) ** 2
    raw = float(candidates[int(np.argmin(objective))])
    steer = float(np.clip(base + gate * (raw - base), -1.0, 1.0))
    return {
        "steer": steer,
        "delta": steer - base,
        "raw_steer": raw,
        "gate": gate,
        "desired_yaw_rate": desired,
        "predicted_base_yaw_rate": float(model.predict_numpy(speed, yaw_rate, base)),
        "predicted_adapted_yaw_rate": float(model.predict_numpy(speed, yaw_rate, steer)),
        "heading_error": heading,
    }


def bounded_inverse_steer_v2(
    model: SpeedConditionedYawDynamics,
    calibrator: BoundedYawRateCalibrator,
    checkpoints: np.ndarray,
    speed: float,
    yaw_rate: float,
    base_steer: float,
    target_speed: float,
    brake: bool,
    spec: InverseDynamicsV2Spec,
) -> Dict[str, float]:
    """Phase- and risk-gated inverse dynamics around the frozen TF++ PID."""
    phase = checkpoint_curvature_phase(checkpoints, spec.phase_curvature_scale)
    raw_desired = float(np.clip(float(speed) * phase["curvature"], -spec.maximum_yaw_rate, spec.maximum_yaw_rate))
    gain = calibrator.predict_numpy(speed, phase["curvature"], phase["entry"], phase["exit"])
    calibrated_desired = float(np.clip(raw_desired * gain, -spec.maximum_yaw_rate, spec.maximum_yaw_rate))

    width = max(spec.full_turn_threshold_rad - spec.turn_threshold_rad, 1e-6)
    turn_gate = float(np.clip((abs(phase["heading_error"]) - spec.turn_threshold_rad) / width, 0.0, 1.0))
    if float(speed) < spec.minimum_speed:
        turn_gate = 0.0
    phase_gate = float(1.0 - (1.0 - spec.exit_gate_floor) * phase["exit"])
    risk_gate = float(not bool(brake) and float(target_speed) >= spec.minimum_target_speed)
    same_direction = calibrated_desired * float(yaw_rate) > 0.0
    overshoot = (
        abs(calibrated_desired) >= spec.overshoot_minimum_yaw_rate
        and same_direction
        and abs(float(yaw_rate)) >= spec.overshoot_ratio * abs(calibrated_desired)
    )
    overshoot_gate = 0.0 if overshoot else 1.0
    combined_gate = turn_gate * phase_gate * risk_gate * overshoot_gate

    base = float(np.clip(base_steer, -1.0, 1.0))
    lower = max(-1.0, base - spec.max_delta)
    upper = min(1.0, base + spec.max_delta)
    candidates = np.linspace(lower, upper, max(int(spec.grid_size), 3), dtype=np.float64)
    predicted = model.predict_numpy(speed, yaw_rate, candidates)
    scale = max(spec.max_delta, 1e-6)
    objective = (predicted - calibrated_desired) ** 2 + spec.base_regularization * ((candidates - base) / scale) ** 2
    raw_steer = float(candidates[int(np.argmin(objective))])
    steer = float(np.clip(base + combined_gate * (raw_steer - base), -1.0, 1.0))
    return {
        "steer": steer,
        "delta": steer - base,
        "raw_steer": raw_steer,
        "gate": combined_gate,
        "turn_gate": turn_gate,
        "phase_gate": phase_gate,
        "risk_gate": risk_gate,
        "overshoot_gate": overshoot_gate,
        "calibration_gain": gain,
        "raw_desired_yaw_rate": raw_desired,
        "desired_yaw_rate": calibrated_desired,
        "predicted_base_yaw_rate": float(model.predict_numpy(speed, yaw_rate, base)),
        "predicted_adapted_yaw_rate": float(model.predict_numpy(speed, yaw_rate, steer)),
        "heading_error": phase["heading_error"],
        "curvature": phase["curvature"],
        "near_curvature": phase["near_curvature"],
        "entry_strength": phase["entry"],
        "exit_strength": phase["exit"],
        "brake": float(bool(brake)),
        "target_speed": float(target_speed),
        "yaw_rate": float(yaw_rate),
        "base_steer": base,
    }


def checkpoint_payload(model: SpeedConditionedYawDynamics, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "vehicle_dynamics_state": model.state_dict(),
        "metadata": {"vehicle_dynamics": metadata},
    }


def calibrator_checkpoint_payload(model: BoundedYawRateCalibrator, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "yaw_rate_calibrator_state": model.state_dict(),
        "metadata": {"yaw_rate_calibrator": metadata},
    }
