"""Small sensor-only safety rules for released TF++ BEV detections.

The released TF++ bounding-box head emits rows with the schema
``[x, y, extent_x, extent_y, yaw, speed, brake, class, score]`` in the
ego frame.  These helpers deliberately do not consume CARLA actors or other
privileged simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Iterable, Sequence


@dataclass(frozen=True)
class OncomingBoxGateConfig:
    """Geometry for an oncoming vehicle occupying a left-turn conflict lane."""

    x_min_m: float = 0.0
    x_max_m: float = 35.0
    y_min_m: float = -4.5
    y_max_m: float = 0.5
    min_abs_yaw_rad: float = 2.30
    min_score: float = 0.50
    vehicle_classes: tuple[int, ...] = (0, 4)


@dataclass
class LeftTurnConflictLatch:
    """Latch only a conflict that was first observed during an active left turn.

    Once triggered, the latch remains active while the vehicle is still
    detected.  A short clear hold bridges detector flicker.  A later oncoming
    vehicle cannot retrigger the latch on a straight road.
    """

    clear_hold_frames: int = 8
    active: bool = False
    clear_remaining: int = 0

    def update(self, *, left_turn_now: bool, conflict_now: bool) -> tuple[bool, bool, int]:
        triggered = False
        if not self.active and left_turn_now and conflict_now:
            self.active = True
            triggered = True

        if self.active:
            if conflict_now:
                self.clear_remaining = max(0, int(self.clear_hold_frames))
            elif self.clear_remaining > 0:
                self.clear_remaining -= 1
            else:
                self.active = False

        return triggered, self.active, self.clear_remaining


def is_oncoming_conflict_box(
    box: Sequence[float],
    config: OncomingBoxGateConfig = OncomingBoxGateConfig(),
) -> bool:
    """Return whether a TF++ detection is an oncoming conflict vehicle."""

    if len(box) < 9:
        return False
    x, y, yaw = float(box[0]), float(box[1]), float(box[4])
    class_id, score = int(round(float(box[7]))), float(box[8])
    wrapped_yaw = (yaw + pi) % (2.0 * pi) - pi
    return (
        class_id in config.vehicle_classes
        and score >= config.min_score
        and config.x_min_m <= x <= config.x_max_m
        and config.y_min_m <= y <= config.y_max_m
        and abs(wrapped_yaw) >= config.min_abs_yaw_rad
    )


def find_oncoming_conflict_boxes(
    boxes: Iterable[Sequence[float]],
    config: OncomingBoxGateConfig = OncomingBoxGateConfig(),
) -> list[Sequence[float]]:
    """Filter detections without changing their order or representation."""

    return [box for box in boxes if is_oncoming_conflict_box(box, config)]


def should_trigger_left_turn_stop_extension(
    *,
    steer: float,
    target_speed_mps: float,
    has_conflict: bool,
    left_steer_threshold: float = -0.10,
    max_target_speed_mps: float = 0.5,
) -> bool:
    """Allow extending an existing stop, never inventing a mid-turn stop."""

    return (
        bool(has_conflict)
        and float(steer) <= float(left_steer_threshold)
        and float(target_speed_mps) <= float(max_target_speed_mps)
    )
