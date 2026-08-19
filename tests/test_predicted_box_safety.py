from math import pi

from teach2drive_adapter.predicted_box_safety import (
    LeftTurnConflictLatch,
    OncomingBoxGateConfig,
    find_oncoming_conflict_boxes,
    is_oncoming_conflict_box,
    should_trigger_left_turn_stop_extension,
)


def box(x=20.0, y=-2.5, yaw=pi, class_id=0, score=0.9):
    return [x, y, 2.4, 1.0, yaw, 0.0, 0.0, class_id, score]


def test_accepts_oncoming_car_in_left_turn_conflict_lane():
    assert is_oncoming_conflict_box(box())
    assert is_oncoming_conflict_box(box(yaw=-pi + 0.05, class_id=4))


def test_rejects_non_vehicle_and_non_oncoming_detections():
    assert not is_oncoming_conflict_box(box(class_id=2))
    assert not is_oncoming_conflict_box(box(yaw=0.0))
    assert not is_oncoming_conflict_box(box(score=0.49))


def test_rejects_boxes_outside_narrow_conflict_region():
    assert not is_oncoming_conflict_box(box(x=-0.01))
    assert not is_oncoming_conflict_box(box(x=35.01))
    assert not is_oncoming_conflict_box(box(y=-4.51))
    assert not is_oncoming_conflict_box(box(y=0.51))


def test_filter_preserves_matching_boxes_and_order():
    first = box(x=30.0)
    second = box(x=10.0, class_id=4)
    result = find_oncoming_conflict_boxes([first, box(class_id=2), second])
    assert result == [first, second]


def test_custom_range_is_honored():
    config = OncomingBoxGateConfig(x_max_m=10.0)
    assert is_oncoming_conflict_box(box(x=9.0), config)
    assert not is_oncoming_conflict_box(box(x=11.0), config)


def test_conflict_latch_requires_left_turn_for_initial_trigger():
    latch = LeftTurnConflictLatch(clear_hold_frames=2)
    assert latch.update(left_turn_now=False, conflict_now=True) == (False, False, 0)
    assert latch.update(left_turn_now=True, conflict_now=True) == (True, True, 2)


def test_active_latch_tracks_conflict_without_stale_turn_memory():
    latch = LeftTurnConflictLatch(clear_hold_frames=2)
    latch.update(left_turn_now=True, conflict_now=True)
    assert latch.update(left_turn_now=False, conflict_now=True) == (False, True, 2)
    assert latch.update(left_turn_now=False, conflict_now=False) == (False, True, 1)
    assert latch.update(left_turn_now=False, conflict_now=False) == (False, True, 0)
    assert latch.update(left_turn_now=False, conflict_now=False) == (False, False, 0)
    # A new straight-road vehicle must not reactivate the cleared latch.
    assert latch.update(left_turn_now=False, conflict_now=True) == (False, False, 0)


def test_stop_extension_trigger_rejects_a_new_mid_turn_stop():
    common = dict(steer=-0.22, has_conflict=True)
    assert not should_trigger_left_turn_stop_extension(
        target_speed_mps=10.0, **common
    )
    assert should_trigger_left_turn_stop_extension(
        target_speed_mps=0.0, **common
    )
    assert not should_trigger_left_turn_stop_extension(
        steer=0.0, target_speed_mps=0.0, has_conflict=True
    )
