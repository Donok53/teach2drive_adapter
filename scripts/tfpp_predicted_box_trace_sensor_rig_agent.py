#!/usr/bin/env python3
"""Sensor-only TF++ agent that traces the released model's BEV detections.

The driving policy is intentionally unchanged.  This diagnostic wrapper
decodes the bounding-box head that TF++ already evaluates during its forward
pass and records the ensemble NMS result next to the original control inputs.
It is used to decide whether a future TTC safety gate can be implemented from
sensor predictions instead of privileged simulator actors.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tfpp_sensor_rig_agent import SensorRigAgent  # noqa: E402
import transfuser_utils as t_u  # noqa: E402


def get_entry_point() -> str:
    return "PredictedBoxTraceSensorRigAgent"


class PredictedBoxTraceSensorRigAgent(SensorRigAgent):
    """Trace TF++ bounding boxes while preserving the exact base action."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(
            path_to_conf_file,
            route_index=route_index,
            traffic_manager=traffic_manager,
        )
        trace_path = os.environ.get("TFPP_PRED_BOX_TRACE_PATH", "").strip()
        if not trace_path:
            raise ValueError("TFPP_PRED_BOX_TRACE_PATH is required")
        path = Path(trace_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._pred_box_trace_handle = path.open("a", encoding="utf-8", buffering=1)
        atexit.register(self._pred_box_trace_handle.close)
        self._pred_boxes_by_net: dict[int, list] = {}
        self._pred_box_step = 0
        self._patch_box_capture()
        print(
            f"[PredictedBoxTraceSensorRigAgent] trace={path} policy_change=off",
            flush=True,
        )

    def _patch_box_capture(self) -> None:
        for index, net in enumerate(self.nets):
            original_forward = net.forward

            def traced_forward(*args, _original=original_forward, _net=net, _index=index, **kwargs):
                output = _original(*args, **kwargs)
                boxes = []
                if (
                    isinstance(output, (tuple, list))
                    and len(output) > 6
                    and output[6] is not None
                ):
                    with torch.no_grad():
                        boxes = _net.convert_features_to_bb_metric(output[6])
                self._pred_boxes_by_net[int(_index)] = boxes
                return output

            net.forward = traced_forward

        original_control = self.nets[0].control_pid_direct

        def traced_control(pred_checkpoints, pred_target_speed, speed, *args, **kwargs):
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
                else []
            )
            payload = {
                "step": int(self._pred_box_step),
                "ego_speed_mps": speed_scalar,
                "target_speed_mps": target_scalar,
                "checkpoints": checkpoints,
                "control": {
                    "steer": float(steer),
                    "throttle": float(throttle),
                    "brake": float(brake),
                },
                # [x, y, extent_x, extent_y, yaw, speed, brake, class, score]
                "boxes": [[float(value) for value in box] for box in boxes],
            }
            self._pred_box_trace_handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._pred_box_step += 1
            return steer, throttle, brake

        self.nets[0].control_pid_direct = traced_control
