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


@dataclass(frozen=True)
class LeftTurnTTCGateConfig:
    """Conservative activation window for a new anticipatory stop."""

    activation_x_min_m: float = 15.0
    activation_x_max_m: float = 35.0
    y_min_m: float = -3.5
    y_max_m: float = 0.5
    min_abs_yaw_rad: float = 2.30
    min_score: float = 0.50
    assumed_oncoming_speed_mps: float = 10.0
    max_ttc_s: float = 2.0
    left_last_checkpoint_y_max_m: float = -0.5
    vehicle_classes: tuple[int, ...] = (0, 4)


@dataclass(frozen=True)
class RightTurnCrossingGateConfig:
    """Activation/continuation geometry for traffic crossing a right turn."""

    activation_x_min_m: float = 6.0
    activation_x_max_m: float = 15.0
    activation_y_min_m: float = -5.0
    activation_y_max_m: float = -0.3
    conflict_x_min_m: float = 0.0
    conflict_x_max_m: float = 16.0
    conflict_y_min_m: float = -5.0
    conflict_y_max_m: float = 5.5
    min_abs_yaw_rad: float = 0.8
    max_abs_yaw_rad: float = 2.2
    min_score: float = 0.50
    right_last_checkpoint_y_min_m: float = 2.0
    vehicle_classes: tuple[int, ...] = (0, 4)


@dataclass(frozen=True)
class RightTurnOncomingTTCGateConfig:
    """Activation/continuation geometry for oncoming right-turn conflicts."""

    activation_x_min_m: float = 10.0
    activation_x_max_m: float = 25.0
    activation_y_min_m: float = -1.0
    activation_y_max_m: float = 5.0
    conflict_x_min_m: float = 0.0
    conflict_x_max_m: float = 25.0
    conflict_y_min_m: float = -4.0
    conflict_y_max_m: float = 5.0
    min_abs_yaw_rad: float = 2.2
    min_score: float = 0.50
    assumed_oncoming_speed_mps: float = 10.0
    max_ttc_s: float = 1.5
    right_last_checkpoint_y_min_m: float = 2.0
    vehicle_classes: tuple[int, ...] = (0, 4)


@dataclass
class OncomingStopExtensionLatch:
    """Latch a conflict first observed during an existing zero-speed decision.

    Once triggered, the latch remains active while the vehicle is still
    detected.  A short clear hold bridges detector flicker.  A later oncoming
    vehicle cannot retrigger the latch unless TF++ again selects zero speed.
    """

    clear_hold_frames: int = 8
    active: bool = False
    clear_remaining: int = 0

    def update(self, *, trigger_now: bool, conflict_now: bool) -> tuple[bool, bool, int]:
        triggered = False
        if not self.active and trigger_now and conflict_now:
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


def find_left_turn_ttc_trigger_boxes(
    boxes: Iterable[Sequence[float]],
    checkpoints: Sequence[Sequence[float]],
    ego_speed_mps: float,
    config: LeftTurnTTCGateConfig = LeftTurnTTCGateConfig(),
) -> list[Sequence[float]]:
    """Find early oncoming threats while the predicted path bends left.

    Activation is intentionally allowed only at a moderate distance. A box
    first noticed inside that window is too late for a new hard stop and is
    rejected, preventing the mid-turn regressions seen in preservation routes.
    """

    if not checkpoints or len(checkpoints[-1]) < 2:
        return []
    if float(checkpoints[-1][1]) > config.left_last_checkpoint_y_max_m:
        return []

    closing_speed = max(
        0.1,
        max(0.0, float(ego_speed_mps)) + config.assumed_oncoming_speed_mps,
    )
    result: list[Sequence[float]] = []
    for box in boxes:
        if len(box) < 9:
            continue
        x, y, yaw = float(box[0]), float(box[1]), float(box[4])
        class_id, score = int(round(float(box[7]))), float(box[8])
        wrapped_yaw = (yaw + pi) % (2.0 * pi) - pi
        if (
            class_id in config.vehicle_classes
            and score >= config.min_score
            and config.activation_x_min_m <= x <= config.activation_x_max_m
            and config.y_min_m <= y <= config.y_max_m
            and abs(wrapped_yaw) >= config.min_abs_yaw_rad
            and x / closing_speed <= config.max_ttc_s
        ):
            result.append(box)
    return result


def _is_right_turn(
    checkpoints: Sequence[Sequence[float]],
    minimum_last_y_m: float,
) -> bool:
    return bool(
        checkpoints
        and len(checkpoints[-1]) >= 2
        and float(checkpoints[-1][1]) >= float(minimum_last_y_m)
    )


def find_right_turn_crossing_boxes(
    boxes: Iterable[Sequence[float]],
    checkpoints: Sequence[Sequence[float]],
    config: RightTurnCrossingGateConfig = RightTurnCrossingGateConfig(),
    *,
    activation: bool,
) -> list[Sequence[float]]:
    """Find lateral traffic crossing the predicted right-turn corridor."""

    if not _is_right_turn(checkpoints, config.right_last_checkpoint_y_min_m):
        return []
    x_min = config.activation_x_min_m if activation else config.conflict_x_min_m
    x_max = config.activation_x_max_m if activation else config.conflict_x_max_m
    y_min = config.activation_y_min_m if activation else config.conflict_y_min_m
    y_max = config.activation_y_max_m if activation else config.conflict_y_max_m
    result: list[Sequence[float]] = []
    for box in boxes:
        if len(box) < 9:
            continue
        x, y, yaw = float(box[0]), float(box[1]), float(box[4])
        class_id, score = int(round(float(box[7]))), float(box[8])
        wrapped_yaw = (yaw + pi) % (2.0 * pi) - pi
        abs_yaw = abs(wrapped_yaw)
        if (
            class_id in config.vehicle_classes
            and score >= config.min_score
            and x_min <= x <= x_max
            and y_min <= y <= y_max
            and config.min_abs_yaw_rad <= abs_yaw <= config.max_abs_yaw_rad
        ):
            result.append(box)
    return result


def find_right_turn_oncoming_ttc_boxes(
    boxes: Iterable[Sequence[float]],
    checkpoints: Sequence[Sequence[float]],
    ego_speed_mps: float,
    config: RightTurnOncomingTTCGateConfig = RightTurnOncomingTTCGateConfig(),
    *,
    activation: bool,
) -> list[Sequence[float]]:
    """Find oncoming traffic intersecting a predicted right-turn corridor."""

    if not _is_right_turn(checkpoints, config.right_last_checkpoint_y_min_m):
        return []
    x_min = config.activation_x_min_m if activation else config.conflict_x_min_m
    x_max = config.activation_x_max_m if activation else config.conflict_x_max_m
    y_min = config.activation_y_min_m if activation else config.conflict_y_min_m
    y_max = config.activation_y_max_m if activation else config.conflict_y_max_m
    closing_speed = max(
        0.1,
        max(0.0, float(ego_speed_mps)) + config.assumed_oncoming_speed_mps,
    )
    result: list[Sequence[float]] = []
    for box in boxes:
        if len(box) < 9:
            continue
        x, y, yaw = float(box[0]), float(box[1]), float(box[4])
        class_id, score = int(round(float(box[7]))), float(box[8])
        wrapped_yaw = (yaw + pi) % (2.0 * pi) - pi
        if (
            class_id in config.vehicle_classes
            and score >= config.min_score
            and x_min <= x <= x_max
            and y_min <= y <= y_max
            and abs(wrapped_yaw) >= config.min_abs_yaw_rad
            and (not activation or x / closing_speed <= config.max_ttc_s)
        ):
            result.append(box)
    return result


def should_trigger_oncoming_stop_extension(
    *,
    target_speed_mps: float,
    has_conflict: bool,
    max_target_speed_mps: float = 0.5,
) -> bool:
    """Extend a zero-speed decision; never invent a new moving stop."""

    return (
        bool(has_conflict)
        and float(target_speed_mps) <= float(max_target_speed_mps)
    )


def is_left_route_command(command: int | float, left_command_value: int = 1) -> bool:
    """Match CARLA Leaderboard's standard RoadOption.LEFT route command."""

    return int(command) == int(left_command_value)
