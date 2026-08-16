import numpy as np

from teach2drive_adapter.data import _future_ego_spatial_checkpoints


def _frame(time_s: float, x: float, y: float, yaw: float = 0.0):
    return {"time": time_s, "odom": {"x": x, "y": y, "yaw": yaw}}


def test_future_ego_path_is_resampled_by_arc_length():
    frames = [_frame(i * 0.25, float(i), 0.1 * float(i)) for i in range(16)]
    target, valid = _future_ego_spatial_checkpoints(frames, 0, 10, max_horizon_s=6.0)

    assert valid == 1.0
    travelled = np.linalg.norm(target, axis=1)
    np.testing.assert_allclose(travelled, 2.5 + np.arange(10), atol=1e-4)
    assert np.all(target[:, 1] > 0.0)


def test_future_ego_path_is_expressed_in_current_ego_frame():
    # yaw=pi/2 means +world-y is straight ahead in the current ego frame.
    frames = [_frame(i * 0.25, 10.0, 20.0 + float(i), np.pi / 2.0) for i in range(16)]
    target, valid = _future_ego_spatial_checkpoints(frames, 0, 10, max_horizon_s=6.0)

    assert valid == 1.0
    np.testing.assert_allclose(target[:, 0], 2.5 + np.arange(10), atol=1e-4)
    np.testing.assert_allclose(target[:, 1], 0.0, atol=1e-4)


def test_future_ego_path_does_not_extrapolate_short_motion():
    frames = [_frame(i * 0.25, 0.25 * float(i), 0.0) for i in range(12)]
    target, valid = _future_ego_spatial_checkpoints(frames, 0, 10, max_horizon_s=6.0)

    assert valid == 0.0
    np.testing.assert_array_equal(target, np.zeros((10, 2), dtype=np.float32))
