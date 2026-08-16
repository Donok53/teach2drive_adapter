import numpy as np
import torch

from teach2drive_adapter.vehicle_dynamics import (
    BoundedYawRateCalibrator,
    InverseDynamicsSpec,
    InverseDynamicsV2Spec,
    SpeedConditionedYawDynamics,
    bounded_inverse_steer,
    bounded_inverse_steer_v2,
    checkpoint_curvature_phase,
    checkpoint_desired_yaw_rate,
)


def test_feature_equivalence_numpy_and_torch():
    coefficients = np.asarray([0.5, 0.2, 0.01, 0.15, -0.05, 0.1, -0.2], dtype=np.float32)
    model = SpeedConditionedYawDynamics(coefficients)
    expected = model.predict_numpy(5.0, 0.1, 0.2)
    actual = model(torch.tensor([5.0]), torch.tensor([0.1]), torch.tensor([0.2])).item()
    assert np.isclose(expected, actual, atol=1e-6)


def test_straight_and_low_speed_preserve_base_steer():
    model = SpeedConditionedYawDynamics(np.asarray([0, 0, 0, 0.2, 0, 0, 0], dtype=np.float32))
    spec = InverseDynamicsSpec()
    straight = np.asarray([[2.0, 0.0], [10.0, 0.0]])
    result = bounded_inverse_steer(model, straight, 5.0, 0.0, 0.2, spec)
    assert result["steer"] == 0.2
    turning = np.asarray([[2.0, 1.0], [8.0, 4.0]])
    result = bounded_inverse_steer(model, turning, 0.5, 0.0, 0.2, spec)
    assert result["steer"] == 0.2


def test_inverse_is_bounded_and_reduces_yaw_error():
    model = SpeedConditionedYawDynamics(np.asarray([0.5, 0, 0, 0.2, 0, 0, 0], dtype=np.float32))
    spec = InverseDynamicsSpec(max_delta=0.12, base_regularization=0.0)
    checkpoints = np.asarray([[2.0, 1.0], [8.0, 4.0]])
    desired, _ = checkpoint_desired_yaw_rate(checkpoints, 5.0)
    result = bounded_inverse_steer(model, checkpoints, 5.0, 0.0, 0.0, spec)
    assert abs(result["delta"]) <= 0.120001
    assert abs(result["predicted_adapted_yaw_rate"] - desired) < abs(result["predicted_base_yaw_rate"] - desired)


def test_checkpoint_phase_detects_entry_and_exit():
    x = np.arange(1.0, 11.0)
    entry = np.column_stack((x, [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50, 1.20, 2.50, 4.00]))
    exit_path = np.column_stack((x, [0.30, 1.00, 2.00, 3.00, 4.00, 3.00, 2.00, 1.00, 0.30, 0.10]))
    entry_phase = checkpoint_curvature_phase(entry)
    exit_phase = checkpoint_curvature_phase(exit_path)
    assert entry_phase["entry"] > entry_phase["exit"]
    assert exit_phase["exit"] > exit_phase["entry"]


def test_v2_risk_and_overshoot_gates_preserve_base():
    dynamics = SpeedConditionedYawDynamics(np.asarray([0.5, 0, 0, 0.2, 0, 0, 0], dtype=np.float32))
    calibrator = BoundedYawRateCalibrator()
    checkpoints = np.asarray([[2.0, 0.1], [4.0, 0.4], [6.0, 1.2], [8.0, 2.4], [10.0, 4.0]])
    spec = InverseDynamicsV2Spec(max_delta=0.12, base_regularization=0.0)
    braking = bounded_inverse_steer_v2(dynamics, calibrator, checkpoints, 5.0, 0.0, 0.1, 4.0, True, spec)
    low_target = bounded_inverse_steer_v2(dynamics, calibrator, checkpoints, 5.0, 0.0, 0.1, 1.0, False, spec)
    assert braking["steer"] == 0.1 and braking["risk_gate"] == 0.0
    assert low_target["steer"] == 0.1 and low_target["risk_gate"] == 0.0
    desired = bounded_inverse_steer_v2(dynamics, calibrator, checkpoints, 5.0, 1.0, 0.1, 4.0, False, spec)
    assert desired["steer"] == 0.1 and desired["overshoot_gate"] == 0.0
