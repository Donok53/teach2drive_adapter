#!/usr/bin/env python3
"""TF++ sensor-rig agent with narrowly gated sensor-only safety shields.

The base planner, target-speed prediction, steering PID, and sensor rig remain
unchanged. The wrapper can extend an existing zero-speed decision, or start an
early stop when the predicted checkpoint path already bends left and an
oncoming detection has a short conservative TTC. Late mid-turn activation is
forbidden. No CARLA actor or future ground-truth information is used.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from teach2drive_adapter.predicted_box_safety import (  # noqa: E402
    LeftTurnTTCGateConfig,
    OncomingStopExtensionLatch,
    OncomingBoxGateConfig,
    RightTurnCrossingGateConfig,
    RightTurnOncomingTTCGateConfig,
    find_left_turn_ttc_trigger_boxes,
    find_oncoming_conflict_boxes,
    find_right_turn_crossing_boxes,
    find_right_turn_oncoming_ttc_boxes,
    should_trigger_oncoming_stop_extension,
)
from tfpp_sensor_rig_agent import SensorRigAgent  # noqa: E402
import transfuser_utils as t_u  # noqa: E402


def get_entry_point() -> str:
    return "PredictedBoxSafetySensorRigAgent"


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)) or default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)) or default)


class PredictedBoxSafetyMixin:
    """Add the narrow TF++ box safety shield after another agent is set up."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(
            path_to_conf_file,
            route_index=route_index,
            traffic_manager=traffic_manager,
        )
        self._box_gate_config = OncomingBoxGateConfig(
            x_min_m=_env_float("TFPP_BOX_SAFETY_X_MIN_M", 0.0),
            x_max_m=_env_float("TFPP_BOX_SAFETY_X_MAX_M", 35.0),
            y_min_m=_env_float("TFPP_BOX_SAFETY_Y_MIN_M", -4.5),
            y_max_m=_env_float("TFPP_BOX_SAFETY_Y_MAX_M", 0.5),
            min_abs_yaw_rad=_env_float("TFPP_BOX_SAFETY_MIN_ABS_YAW_RAD", 2.30),
            min_score=_env_float("TFPP_BOX_SAFETY_MIN_SCORE", 0.50),
        )
        self._ttc_gate_config = LeftTurnTTCGateConfig(
            activation_x_min_m=_env_float("TFPP_BOX_TTC_X_MIN_M", 15.0),
            activation_x_max_m=_env_float("TFPP_BOX_TTC_X_MAX_M", 35.0),
            y_min_m=_env_float("TFPP_BOX_TTC_Y_MIN_M", -3.5),
            y_max_m=_env_float("TFPP_BOX_TTC_Y_MAX_M", 0.5),
            min_abs_yaw_rad=_env_float("TFPP_BOX_TTC_MIN_ABS_YAW_RAD", 2.30),
            min_score=_env_float("TFPP_BOX_TTC_MIN_SCORE", 0.50),
            assumed_oncoming_speed_mps=_env_float(
                "TFPP_BOX_TTC_ASSUMED_ONCOMING_SPEED_MPS", 10.0
            ),
            max_ttc_s=_env_float("TFPP_BOX_TTC_MAX_S", 2.0),
            left_last_checkpoint_y_max_m=_env_float(
                "TFPP_BOX_TTC_LEFT_LAST_CHECKPOINT_Y_MAX_M", -0.5
            ),
        )
        self._left_steer_threshold = _env_float("TFPP_BOX_SAFETY_LEFT_STEER", -0.10)
        self._right_crossing_config = RightTurnCrossingGateConfig()
        self._right_oncoming_config = RightTurnOncomingTTCGateConfig()
        # Right-turn hard stops are experimental and disabled by default. In
        # closed loop they can leave the ego vehicle inside the conflict zone,
        # where repeated traffic retriggers the latch and causes a side impact
        # or route timeout. Keep the proven left-turn shields independent.
        self._right_safety_enabled = bool(_env_int("TFPP_BOX_RIGHT_SAFETY", 0))
        self._max_trigger_target_speed_mps = _env_float(
            "TFPP_BOX_SAFETY_MAX_TRIGGER_TARGET_SPEED_MPS", 0.5
        )
        self._clear_hold_frames = _env_int("TFPP_BOX_SAFETY_CLEAR_HOLD_FRAMES", 8)
        self._stop_extension_latch = OncomingStopExtensionLatch(self._clear_hold_frames)
        self._ttc_latch = OncomingStopExtensionLatch(self._clear_hold_frames)
        self._right_crossing_latch = OncomingStopExtensionLatch(self._clear_hold_frames)
        self._right_oncoming_latch = OncomingStopExtensionLatch(self._clear_hold_frames)
        self._pred_boxes_by_net: dict[int, list] = {}
        self._safety_step = 0

        trace_path = os.environ.get("TFPP_PRED_BOX_TRACE_PATH", "").strip()
        self._safety_trace_handle = None
        if trace_path:
            path = Path(trace_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._safety_trace_handle = path.open("a", encoding="utf-8", buffering=1)
            atexit.register(self._safety_trace_handle.close)

        self._patch_box_safety()
        print(
            "[PredictedBoxSafetySensorRigAgent] sensor_only=on "
            f"left_steer<={self._left_steer_threshold:.3f} "
            f"max_trigger_target_speed={self._max_trigger_target_speed_mps:.3f} "
            f"box_gate={self._box_gate_config} ttc_gate={self._ttc_gate_config}",
            flush=True,
        )

    def _patch_box_safety(self) -> None:
        for index, net in enumerate(self.nets):
            original_forward = net.forward

            def traced_forward(*args, _original=original_forward, _net=net, _index=index, **kwargs):
                output = _original(*args, **kwargs)
                boxes = []
                if isinstance(output, (tuple, list)) and len(output) > 6 and output[6] is not None:
                    with torch.no_grad():
                        boxes = _net.convert_features_to_bb_metric(output[6])
                self._pred_boxes_by_net[int(_index)] = boxes
                return output

            net.forward = traced_forward

        original_control = self.nets[0].control_pid_direct

        def safety_control(pred_checkpoints, pred_target_speed, speed, *args, **kwargs):
            steer, throttle, brake = original_control(
                pred_checkpoints,
                pred_target_speed,
                speed,
                *args,
                **kwargs,
            )
            member_boxes = [
                self._pred_boxes_by_net[index]
                for index in sorted(self._pred_boxes_by_net)
                if self._pred_boxes_by_net[index] is not None
            ]
            boxes = (
                t_u.non_maximum_suppression(member_boxes, self.config.iou_treshold_nms)
                if member_boxes
                else []
            )
            conflicts = find_oncoming_conflict_boxes(boxes, self._box_gate_config)

            left_turn_now = float(steer) <= self._left_steer_threshold
            speed_scalar = (
                float(speed.detach().reshape(-1)[0].item())
                if torch.is_tensor(speed)
                else float(speed)
            )
            target_scalar = (
                float(pred_target_speed.detach().reshape(-1)[0].item())
                if torch.is_tensor(pred_target_speed)
                else float(pred_target_speed)
            )
            checkpoints = (
                pred_checkpoints.detach().float().cpu().reshape(-1, 2).tolist()
                if torch.is_tensor(pred_checkpoints)
                else np.asarray(pred_checkpoints, dtype=np.float32).reshape(-1, 2).tolist()
            )
            # This shield may extend a stop the released TF++ policy already
            # selected, but it must never invent a new mid-turn stop.  The
            # latter caused a large lane excursion in the preservation route.
            base_stop_now = target_scalar <= self._max_trigger_target_speed_mps
            trigger_context = should_trigger_oncoming_stop_extension(
                target_speed_mps=target_scalar,
                has_conflict=bool(conflicts),
                max_target_speed_mps=self._max_trigger_target_speed_mps,
            )
            ttc_trigger_boxes = find_left_turn_ttc_trigger_boxes(
                boxes,
                checkpoints,
                speed_scalar,
                self._ttc_gate_config,
            )
            right_crossing_trigger_boxes = []
            right_crossing_conflicts = []
            right_oncoming_trigger_boxes = []
            right_oncoming_conflicts = []
            if self._right_safety_enabled:
                right_crossing_trigger_boxes = find_right_turn_crossing_boxes(
                    boxes,
                    checkpoints,
                    self._right_crossing_config,
                    activation=True,
                )
                right_crossing_conflicts = find_right_turn_crossing_boxes(
                    boxes,
                    checkpoints,
                    self._right_crossing_config,
                    activation=False,
                )
                right_oncoming_trigger_boxes = find_right_turn_oncoming_ttc_boxes(
                    boxes,
                    checkpoints,
                    speed_scalar,
                    self._right_oncoming_config,
                    activation=True,
                )
                right_oncoming_conflicts = find_right_turn_oncoming_ttc_boxes(
                    boxes,
                    checkpoints,
                    speed_scalar,
                    self._right_oncoming_config,
                    activation=False,
                )
            stop_triggered, stop_active, stop_clear_remaining = (
                self._stop_extension_latch.update(
                    trigger_now=trigger_context,
                    conflict_now=bool(conflicts),
                )
            )
            ttc_triggered, ttc_active, ttc_clear_remaining = self._ttc_latch.update(
                trigger_now=bool(ttc_trigger_boxes),
                conflict_now=bool(conflicts),
            )
            crossing_triggered, crossing_active, crossing_clear_remaining = (
                self._right_crossing_latch.update(
                    trigger_now=bool(right_crossing_trigger_boxes),
                    conflict_now=bool(right_crossing_conflicts),
                )
            )
            right_oncoming_triggered, right_oncoming_active, right_oncoming_clear_remaining = (
                self._right_oncoming_latch.update(
                    trigger_now=bool(right_oncoming_trigger_boxes),
                    conflict_now=bool(right_oncoming_conflicts),
                )
            )
            triggered = (
                stop_triggered
                or ttc_triggered
                or crossing_triggered
                or right_oncoming_triggered
            )
            applied = (
                stop_active or ttc_active or crossing_active or right_oncoming_active
            )
            clear_remaining = max(
                stop_clear_remaining,
                ttc_clear_remaining,
                crossing_clear_remaining,
                right_oncoming_clear_remaining,
            )
            base_control = (float(steer), float(throttle), float(brake))
            if applied:
                throttle, brake = 0.0, 1.0

            if self._safety_trace_handle is not None:
                payload = {
                    "step": int(self._safety_step),
                    "ego_speed_mps": speed_scalar,
                    "target_speed_mps": target_scalar,
                    "checkpoints": checkpoints,
                    "base_control": {
                        "steer": base_control[0],
                        "throttle": base_control[1],
                        "brake": base_control[2],
                    },
                    "control": {
                        "steer": float(steer),
                        "throttle": float(throttle),
                        "brake": float(brake),
                    },
                    "left_turn_now": bool(left_turn_now),
                    "base_stop_now": bool(base_stop_now),
                    "trigger_context": bool(trigger_context),
                    "triggered": bool(triggered),
                    "stop_triggered": bool(stop_triggered),
                    "stop_active": bool(stop_active),
                    "ttc_triggered": bool(ttc_triggered),
                    "ttc_active": bool(ttc_active),
                    "right_crossing_triggered": bool(crossing_triggered),
                    "right_crossing_active": bool(crossing_active),
                    "right_oncoming_triggered": bool(right_oncoming_triggered),
                    "right_oncoming_active": bool(right_oncoming_active),
                    "clear_hold_remaining": int(clear_remaining),
                    "hazard": bool(conflicts),
                    "applied": bool(applied),
                    "conflicts": [[float(value) for value in box] for box in conflicts],
                    "ttc_trigger_boxes": [
                        [float(value) for value in box] for box in ttc_trigger_boxes
                    ],
                    "right_crossing_trigger_boxes": [
                        [float(value) for value in box]
                        for box in right_crossing_trigger_boxes
                    ],
                    "right_oncoming_trigger_boxes": [
                        [float(value) for value in box]
                        for box in right_oncoming_trigger_boxes
                    ],
                    "boxes": [[float(value) for value in box] for box in boxes],
                }
                self._safety_trace_handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._safety_step += 1
            return steer, throttle, brake

        self.nets[0].control_pid_direct = safety_control


class PredictedBoxSafetySensorRigAgent(PredictedBoxSafetyMixin, SensorRigAgent):
    """Preserve released TF++ actions except for narrowly detected conflicts."""
