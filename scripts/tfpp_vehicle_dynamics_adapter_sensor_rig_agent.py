#!/usr/bin/env python3
"""TF++ agent with a bounded, planner-independent vehicle dynamics inverse."""

from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tfpp_sensor_rig_agent import SensorRigAgent  # noqa: E402


def get_entry_point() -> str:
    return "VehicleDynamicsAdapterSensorRigAgent"


class VehicleDynamicsAdapterSensorRigAgent(SensorRigAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)
        adapter_root = Path(os.environ.get("ADAPTER_ROOT", Path(__file__).resolve().parents[1])).expanduser()
        if str(adapter_root) not in sys.path:
            sys.path.insert(0, str(adapter_root))
        from teach2drive_adapter.vehicle_dynamics import InverseDynamicsSpec, SpeedConditionedYawDynamics

        checkpoint_path = Path(os.environ["TFPP_VEHICLE_DYNAMICS_CHECKPOINT"]).expanduser()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self._vehicle_dynamics = SpeedConditionedYawDynamics()
        self._vehicle_dynamics.load_state_dict(checkpoint["vehicle_dynamics_state"])
        self._vehicle_dynamics.eval()
        self._inverse_spec = InverseDynamicsSpec(
            max_delta=float(os.environ.get("TFPP_DYNAMICS_MAX_DELTA", "0.12")),
            minimum_speed=float(os.environ.get("TFPP_DYNAMICS_MINIMUM_SPEED", "1.0")),
            turn_threshold_rad=float(os.environ.get("TFPP_DYNAMICS_TURN_THRESHOLD", "0.04")),
            full_turn_threshold_rad=float(os.environ.get("TFPP_DYNAMICS_FULL_TURN_THRESHOLD", "0.12")),
            maximum_yaw_rate=float(os.environ.get("TFPP_DYNAMICS_MAXIMUM_YAW_RATE", "0.80")),
            base_regularization=float(os.environ.get("TFPP_DYNAMICS_BASE_REGULARIZATION", "0.01")),
        )
        self._dynamics_blend = float(os.environ.get("TFPP_DYNAMICS_BLEND", "1.0"))
        self._yaw_rates: deque[float] = deque(maxlen=int(os.environ.get("TFPP_DYNAMICS_YAW_WINDOW", "11")))
        self._vehicle_yaw_rate = 0.0
        self._patch_lateral_controllers()
        metadata = checkpoint.get("metadata", {}).get("vehicle_dynamics", {})
        print(
            "[VehicleDynamicsAdapter] loaded "
            f"checkpoint={checkpoint_path} blend={self._dynamics_blend:.3f} "
            f"max_delta={self._inverse_spec.max_delta:.3f} val_r2={metadata.get('val_metrics', {}).get('r2', 'n/a')}",
            flush=True,
        )

    def _patch_lateral_controllers(self) -> None:
        from teach2drive_adapter.vehicle_dynamics import bounded_inverse_steer

        for net in self.nets:
            original = net.control_pid_direct

            def adapted_control(pred_checkpoints, pred_target_speed, speed, *args, _original=original, **kwargs):
                steer, throttle, brake = _original(pred_checkpoints, pred_target_speed, speed, *args, **kwargs)
                speed_value = float(speed.detach().reshape(-1)[0].item()) if torch.is_tensor(speed) else float(np.asarray(speed).reshape(-1)[0])
                result = bounded_inverse_steer(
                    self._vehicle_dynamics,
                    pred_checkpoints,
                    speed_value,
                    self._vehicle_yaw_rate,
                    float(steer),
                    self._inverse_spec,
                )
                adapted = float(steer) + self._dynamics_blend * result["delta"]
                return float(np.clip(adapted, -1.0, 1.0)), throttle, brake

            net.control_pid_direct = adapted_control

    def run_step(self, input_data, timestamp, sensors=None):
        packet = input_data.get("imu") if isinstance(input_data, dict) else None
        if packet is not None:
            imu = np.asarray(packet[1], dtype=np.float64).reshape(-1)
            if imu.size >= 6 and np.isfinite(imu[5]):
                self._yaw_rates.append(float(imu[5]))
                self._vehicle_yaw_rate = float(np.mean(self._yaw_rates))
        return super().run_step(input_data, timestamp, sensors=sensors)
