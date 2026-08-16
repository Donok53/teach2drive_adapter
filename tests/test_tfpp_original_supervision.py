import numpy as np

from teach2drive_adapter.data import (
    TFPP_TARGET_SPEEDS,
    _tfpp_command_one_hot,
    _tfpp_two_hot_target_speed,
)


def test_tfpp_two_hot_interpolates_direct_target_speed():
    label = _tfpp_two_hot_target_speed(6.0, False)
    assert label.shape == (len(TFPP_TARGET_SPEEDS),)
    np.testing.assert_allclose(label, [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.isclose(label.sum(), 1.0)


def test_tfpp_two_hot_brake_forces_zero_class():
    label = _tfpp_two_hot_target_speed(8.0, True)
    np.testing.assert_allclose(label, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_tfpp_command_matches_carla_command_indexing():
    np.testing.assert_allclose(_tfpp_command_one_hot(1), [1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(_tfpp_command_one_hot(4), [0, 0, 0, 1, 0, 0])
    np.testing.assert_allclose(_tfpp_command_one_hot(None), [0, 0, 0, 1, 0, 0])
