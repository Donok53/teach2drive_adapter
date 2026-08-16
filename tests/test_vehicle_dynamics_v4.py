import numpy as np
import torch

from teach2drive_adapter.vehicle_dynamics_v3 import LaggedSpeedConditionedYawDynamics
from teach2drive_adapter.vehicle_dynamics_v4 import (
    HorizonYawRateCalibrator,
    InverseDynamicsV4Spec,
    bounded_horizon_inverse_steer_v4,
    checkpoint_pure_pursuit_sequence,
)


def test_horizon_calibrator_starts_at_identity():
    model = HorizonYawRateCalibrator()
    gain = model.predict_numpy(5.0, np.asarray([0.03, 0.05, 0.08]), 1.0, 0.0)
    assert np.allclose(gain, 1.0, atol=1e-6)


def test_checkpoint_target_has_entry_phase():
    x = np.arange(1.0, 11.0)
    path = np.column_stack((x, 0.003 * x**3))
    target = checkpoint_pure_pursuit_sequence(path, 5.0)
    assert target["entry"] > target["exit"]
    assert target["curvature"][-1] > target["curvature"][0]


def test_v4_hard_safety_gates_preserve_base():
    coefficients = np.zeros(11, dtype=np.float32)
    coefficients[3] = 0.2
    dynamics = LaggedSpeedConditionedYawDynamics(coefficients)
    calibrator = HorizonYawRateCalibrator()
    x = np.arange(1.0, 11.0)
    checkpoints = np.column_stack((x, 0.04 * x**2))
    spec = InverseDynamicsV4Spec(base_regularization=0.0)
    braking = bounded_horizon_inverse_steer_v4(
        dynamics, calibrator, checkpoints, 5.0, 0.0, 0.1, 0.0, 0.0, 5.0, True, spec
    )
    low_speed = bounded_horizon_inverse_steer_v4(
        dynamics, calibrator, checkpoints, 5.0, 0.0, 0.1, 0.0, 0.0, 1.0, False, spec
    )
    assert braking["risk_gate"] == 0.0 and braking["steer"] == 0.1
    assert low_speed["risk_gate"] == 0.0 and low_speed["steer"] == 0.1


def test_v4_inverse_remains_bounded():
    coefficients = np.zeros(11, dtype=np.float32)
    coefficients[0] = 0.5
    coefficients[3] = 0.2
    dynamics = LaggedSpeedConditionedYawDynamics(coefficients)
    calibrator = HorizonYawRateCalibrator()
    x = np.arange(1.0, 11.0)
    checkpoints = np.column_stack((x, 0.04 * x**2))
    result = bounded_horizon_inverse_steer_v4(
        dynamics, calibrator, checkpoints, 5.0, 0.0, 0.0, 0.0, 0.0, 5.0, False, InverseDynamicsV4Spec()
    )
    assert abs(result["delta"]) <= 0.120001
    assert len(result["calibration_gain"]) == 3
