#!/usr/bin/env python3
"""TF++ agent with lag-aware, horizon-aligned Tesla yaw dynamics v3."""

from __future__ import annotations

import json
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
    return "VehicleDynamicsV3SensorRigAgent"


class VehicleDynamicsV3SensorRigAgent(SensorRigAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)
        adapter_root = Path(os.environ.get("ADAPTER_ROOT", Path(__file__).resolve().parents[1])).expanduser()
        if str(adapter_root) not in sys.path:
            sys.path.insert(0, str(adapter_root))
        from teach2drive_adapter.vehicle_dynamics_v3 import InverseDynamicsV3Spec, LaggedSpeedConditionedYawDynamics

        checkpoint_path = Path(os.environ["TFPP_VEHICLE_DYNAMICS_V3_CHECKPOINT"]).expanduser()
        payload = torch.load(checkpoint_path, map_location="cpu")
        self._vehicle_dynamics = LaggedSpeedConditionedYawDynamics()
        self._vehicle_dynamics.load_state_dict(payload["lagged_vehicle_dynamics_state"])
        self._vehicle_dynamics.eval()
        self._inverse_spec = InverseDynamicsV3Spec(
            max_delta=float(os.environ.get("TFPP_DYNAMICS_MAX_DELTA", "0.12")),
            minimum_speed=float(os.environ.get("TFPP_DYNAMICS_MINIMUM_SPEED", "1.0")),
            turn_threshold_yaw_rate=float(os.environ.get("TFPP_DYNAMICS_TURN_THRESHOLD_YAW", "0.03")),
            full_turn_threshold_yaw_rate=float(os.environ.get("TFPP_DYNAMICS_FULL_TURN_THRESHOLD_YAW", "0.12")),
            maximum_yaw_rate=float(os.environ.get("TFPP_DYNAMICS_MAXIMUM_YAW_RATE", "0.80")),
            curvature_probe_distance=float(os.environ.get("TFPP_DYNAMICS_CURVATURE_PROBE", "0.75")),
            risk_gate_floor=float(os.environ.get("TFPP_DYNAMICS_RISK_GATE_FLOOR", "0.25")),
            minimum_target_speed=float(os.environ.get("TFPP_DYNAMICS_MINIMUM_TARGET_SPEED", "1.0")),
            base_regularization=float(os.environ.get("TFPP_DYNAMICS_BASE_REGULARIZATION", "0.01")),
        )
        self._dynamics_blend = float(os.environ.get("TFPP_DYNAMICS_BLEND", "0.5"))
        self._lag1_frames = int(os.environ.get("TFPP_DYNAMICS_LAG1_FRAMES", "5"))
        self._lag2_frames = int(os.environ.get("TFPP_DYNAMICS_LAG2_FRAMES", "10"))
        history_size = max(self._lag2_frames + 2, 16)
        self._steer_commands: deque[float] = deque(maxlen=history_size)
        self._yaw_rates: deque[float] = deque(maxlen=int(os.environ.get("TFPP_DYNAMICS_YAW_WINDOW", "11")))
        self._vehicle_yaw_rate = 0.0
        self._vehicle_heading = float("nan")
        self._trace_timestamp = 0.0
        self._trace_step = 0
        env_route_index = os.environ.get("TFPP_DYNAMICS_ROUTE_INDEX", "")
        self._route_index = route_index if route_index is not None else (int(env_route_index) if env_route_index else None)
        self._trace_file = None
        trace_path = os.environ.get("TFPP_DYNAMICS_TRACE_PATH", "")
        if trace_path:
            path = Path(trace_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_file = path.open("a", encoding="utf-8", buffering=1)
        self._patch_lateral_controller()
        metadata = payload.get("metadata", {}).get("lagged_vehicle_dynamics", {})
        print(
            "[VehicleDynamicsV3] loaded "
            f"checkpoint={checkpoint_path} blend={self._dynamics_blend:.3f} "
            f"val_mae={metadata.get('val_metrics', {}).get('mae', 'n/a')} "
            f"lag_frames=[{self._lag1_frames},{self._lag2_frames}] trace={trace_path or 'off'}",
            flush=True,
        )

    def _history_value(self, frames: int, fallback: float) -> float:
        if len(self._steer_commands) >= frames:
            return float(self._steer_commands[-frames])
        if self._steer_commands:
            return float(self._steer_commands[0])
        return float(fallback)

    def _patch_lateral_controller(self) -> None:
        from teach2drive_adapter.vehicle_dynamics_v3 import bounded_horizon_inverse_steer_v3

        original = self.nets[0].control_pid_direct

        def adapted_control(pred_checkpoints, pred_target_speed, speed, *args, **kwargs):
            steer, throttle, brake = original(pred_checkpoints, pred_target_speed, speed, *args, **kwargs)
            speed_value = float(speed.detach().reshape(-1)[0].item()) if torch.is_tensor(speed) else float(np.asarray(speed).reshape(-1)[0])
            target_speed = float(pred_target_speed.detach().reshape(-1)[0].item()) if torch.is_tensor(pred_target_speed) else float(pred_target_speed)
            lag1 = self._history_value(self._lag1_frames, float(steer))
            lag2 = self._history_value(self._lag2_frames, lag1)
            result = bounded_horizon_inverse_steer_v3(
                self._vehicle_dynamics,
                pred_checkpoints,
                speed_value,
                self._vehicle_yaw_rate,
                float(steer),
                lag1,
                lag2,
                target_speed,
                bool(brake),
                self._inverse_spec,
            )
            final_delta = self._dynamics_blend * float(result["delta"])
            adapted = float(np.clip(float(steer) + final_delta, -1.0, 1.0))
            self._steer_commands.append(adapted)
            if self._trace_file is not None:
                record = dict(result)
                record.update(
                    {
                        "timestamp": float(self._trace_timestamp),
                        "trace_step": int(self._trace_step),
                        "route_index": self._route_index,
                        "vehicle_heading": self._vehicle_heading,
                        "blend": self._dynamics_blend,
                        "adapter_delta_pre_blend": float(result["delta"]),
                        "adapter_delta": final_delta,
                        "final_steer": adapted,
                        "throttle": float(throttle),
                    }
                )
                self._trace_file.write(json.dumps(record, sort_keys=True) + "\n")
                self._trace_step += 1
            return adapted, throttle, brake

        self.nets[0].control_pid_direct = adapted_control

    def run_step(self, input_data, timestamp, sensors=None):
        self._trace_timestamp = float(timestamp)
        packet = input_data.get("imu") if isinstance(input_data, dict) else None
        if packet is not None:
            imu = np.asarray(packet[1], dtype=np.float64).reshape(-1)
            if imu.size >= 6 and np.isfinite(imu[5]):
                self._yaw_rates.append(float(imu[5]))
                self._vehicle_yaw_rate = float(np.mean(self._yaw_rates))
            if imu.size >= 7 and np.isfinite(imu[6]):
                self._vehicle_heading = float(imu[6])
        return super().run_step(input_data, timestamp, sensors=sensors)

    def destroy(self, results=None):
        try:
            if self._trace_file is not None:
                self._trace_file.close()
                self._trace_file = None
        finally:
            return super().destroy(results=results)
