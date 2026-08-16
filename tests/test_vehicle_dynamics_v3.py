import numpy as np
import torch

from teach2drive_adapter.vehicle_dynamics_v3 import (
    InverseDynamicsV3Spec,
    LaggedSpeedConditionedYawDynamics,
    bounded_horizon_inverse_steer_v3,
    checkpoint_curvature_sequence,
    rollout_constant_steer,
)


def test_lagged_feature_equivalence_numpy_and_torch():
    coefficients = np.linspace(-0.2, 0.3, 11, dtype=np.float32)
    model = LaggedSpeedConditionedYawDynamics(coefficients)
    expected = model.predict_numpy(5.0, 0.1, 0.2, 0.15, 0.05)
    actual = model(
        torch.tensor([5.0]),
        torch.tensor([0.1]),
        torch.tensor([0.2]),
        torch.tensor([0.15]),
        torch.tensor([0.05]),
    ).item()
    assert np.isclose(expected, actual, atol=1e-6)


def test_checkpoint_curvature_sequence_detects_turn_direction():
    x = np.arange(1.0, 11.0)
    left = np.column_stack((x, 0.04 * x**2))
    right = left * np.asarray([1.0, -1.0])
    straight = np.column_stack((x, np.zeros_like(x)))
    left_result = checkpoint_curvature_sequence(left, 5.0)
    right_result = checkpoint_curvature_sequence(right, 5.0)
    straight_result = checkpoint_curvature_sequence(straight, 5.0)
    assert np.mean(left_result["curvature"]) > 0.0
    assert np.mean(right_result["curvature"]) < 0.0
    assert np.max(np.abs(straight_result["curvature"])) < 1e-8


def test_lag_history_changes_rollout():
    coefficients = np.zeros(11, dtype=np.float32)
    coefficients[8] = 0.5
    model = LaggedSpeedConditionedYawDynamics(coefficients)
    without_lag = rollout_constant_steer(model, 5.0, 0.0, 0.0, 0.0, 0.0, 1)
    with_lag = rollout_constant_steer(model, 5.0, 0.0, 0.0, 0.4, 0.0, 1)
    assert with_lag.item() > without_lag.item()


def test_horizon_inverse_is_bounded_and_improves_tracking():
    coefficients = np.zeros(11, dtype=np.float32)
    coefficients[0] = 0.5
    coefficients[3] = 0.2
    model = LaggedSpeedConditionedYawDynamics(coefficients)
    x = np.arange(1.0, 11.0)
    checkpoints = np.column_stack((x, 0.04 * x**2))
    spec = InverseDynamicsV3Spec(max_delta=0.12, base_regularization=0.0)
    result = bounded_horizon_inverse_steer_v3(model, checkpoints, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0, False, spec)
    desired = np.asarray(result["desired_yaw_rate"])
    base_error = np.mean((np.asarray(result["predicted_base_yaw_rate"]) - desired) ** 2)
    adapted_error = np.mean((np.asarray(result["predicted_adapted_yaw_rate"]) - desired) ** 2)
    assert abs(result["delta"]) <= 0.120001
    assert adapted_error < base_error


def test_risk_gate_uses_floor_instead_of_hard_zero():
    coefficients = np.zeros(11, dtype=np.float32)
    coefficients[3] = 0.2
    model = LaggedSpeedConditionedYawDynamics(coefficients)
    x = np.arange(1.0, 11.0)
    checkpoints = np.column_stack((x, 0.04 * x**2))
    spec = InverseDynamicsV3Spec(risk_gate_floor=0.25, base_regularization=0.0)
    result = bounded_horizon_inverse_steer_v3(model, checkpoints, 5.0, 0.0, 0.0, 0.0, 0.0, 0.5, True, spec)
    assert result["risk_gate"] == 0.25
    assert abs(result["delta"]) > 0.0
