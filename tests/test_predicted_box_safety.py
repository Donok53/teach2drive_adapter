from math import pi

from teach2drive_adapter.predicted_box_safety import (
    LeftTurnTTCGateConfig,
    OncomingStopExtensionLatch,
    OncomingBoxGateConfig,
    RightTurnCrossingGateConfig,
    RightTurnOncomingTTCGateConfig,
    find_left_turn_ttc_trigger_boxes,
    find_oncoming_conflict_boxes,
    find_right_turn_crossing_boxes,
    find_right_turn_oncoming_ttc_boxes,
    is_oncoming_conflict_box,
    is_left_route_command,
    should_trigger_oncoming_stop_extension,
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


def test_conflict_latch_requires_explicit_initial_trigger():
    latch = OncomingStopExtensionLatch(clear_hold_frames=2)
    assert latch.update(trigger_now=False, conflict_now=True) == (False, False, 0)
    assert latch.update(trigger_now=True, conflict_now=True) == (True, True, 2)


def test_active_latch_tracks_conflict_without_stale_turn_memory():
    latch = OncomingStopExtensionLatch(clear_hold_frames=2)
    latch.update(trigger_now=True, conflict_now=True)
    assert latch.update(trigger_now=False, conflict_now=True) == (False, True, 2)
    assert latch.update(trigger_now=False, conflict_now=False) == (False, True, 1)
    assert latch.update(trigger_now=False, conflict_now=False) == (False, True, 0)
    assert latch.update(trigger_now=False, conflict_now=False) == (False, False, 0)
    # A new straight-road vehicle must not reactivate the cleared latch.
    assert latch.update(trigger_now=False, conflict_now=True) == (False, False, 0)


def test_stop_extension_trigger_rejects_a_new_mid_turn_stop():
    assert not should_trigger_oncoming_stop_extension(
        target_speed_mps=10.0, has_conflict=True
    )
    assert should_trigger_oncoming_stop_extension(
        target_speed_mps=0.0, has_conflict=True
    )
    assert not should_trigger_oncoming_stop_extension(
        target_speed_mps=0.0, has_conflict=False
    )


def test_left_route_command_is_explicit_and_does_not_match_right_or_straight():
    assert is_left_route_command(1)
    assert not is_left_route_command(2)
    assert not is_left_route_command(3)
    assert not is_left_route_command(4)


def test_left_turn_ttc_triggers_early_mission13_like_threat():
    checkpoints = [[float(i), -0.07 * i] for i in range(2, 12)]
    threat = box(x=24.6, y=-2.9, yaw=pi)
    assert find_left_turn_ttc_trigger_boxes([threat], checkpoints, 5.7) == [threat]


def test_left_turn_ttc_rejects_late_and_straight_path_stops():
    left = [[float(i), -0.07 * i] for i in range(2, 12)]
    straight = [[float(i), 0.0] for i in range(2, 12)]
    late = box(x=10.0, y=-2.5, yaw=pi)
    early = box(x=24.0, y=-2.5, yaw=pi)
    assert find_left_turn_ttc_trigger_boxes([late], left, 12.0) == []
    assert find_left_turn_ttc_trigger_boxes([early], straight, 6.0) == []


def test_left_turn_ttc_rejects_same_direction_vehicle():
    left = [[float(i), -0.07 * i] for i in range(2, 12)]
    same_direction = box(x=20.0, y=-2.5, yaw=0.0)
    assert find_left_turn_ttc_trigger_boxes([same_direction], left, 6.0) == []


def test_left_turn_ttc_custom_activation_window_is_honored():
    config = LeftTurnTTCGateConfig(activation_x_min_m=20.0)
    left = [[float(i), -0.07 * i] for i in range(2, 12)]
    assert find_left_turn_ttc_trigger_boxes([box(x=19.0)], left, 10.0, config) == []


def test_right_turn_crossing_gate_matches_mission4_activation():
    right = [[float(i), 0.7 * i] for i in range(2, 12)]
    threat = box(x=8.1, y=-3.8, yaw=pi / 2)
    assert find_right_turn_crossing_boxes(
        [threat], right, activation=True
    ) == [threat]


def test_right_turn_crossing_gate_rejects_straight_and_oncoming_boxes():
    right = [[float(i), 0.7 * i] for i in range(2, 12)]
    straight = [[float(i), 0.0] for i in range(2, 12)]
    crossing = box(x=8.0, y=-3.0, yaw=pi / 2)
    oncoming = box(x=8.0, y=-3.0, yaw=pi)
    assert find_right_turn_crossing_boxes([crossing], straight, activation=True) == []
    assert find_right_turn_crossing_boxes([oncoming], right, activation=True) == []


def test_right_turn_crossing_continuation_tracks_vehicle_after_activation_window():
    right = [[float(i), 0.7 * i] for i in range(2, 12)]
    passed_activation_edge = box(x=5.0, y=3.0, yaw=pi / 2)
    assert find_right_turn_crossing_boxes(
        [passed_activation_edge], right, activation=True
    ) == []
    assert find_right_turn_crossing_boxes(
        [passed_activation_edge], right, activation=False
    ) == [passed_activation_edge]


def test_right_turn_oncoming_ttc_matches_mission5_activation():
    right = [[float(i), 0.55 * i] for i in range(2, 12)]
    threat = box(x=11.2, y=4.2, yaw=-2.48)
    assert find_right_turn_oncoming_ttc_boxes(
        [threat], right, 9.4, activation=True
    ) == [threat]


def test_right_turn_oncoming_ttc_rejects_left_path_and_distant_box():
    right = [[float(i), 0.55 * i] for i in range(2, 12)]
    left = [[float(i), -0.55 * i] for i in range(2, 12)]
    threat = box(x=11.2, y=4.2, yaw=-2.48)
    distant = box(x=24.0, y=4.2, yaw=-2.48)
    assert find_right_turn_oncoming_ttc_boxes(
        [threat], left, 9.4, activation=True
    ) == []
    assert find_right_turn_oncoming_ttc_boxes(
        [distant], right, 0.0, activation=True
    ) == []
