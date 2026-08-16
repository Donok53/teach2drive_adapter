#!/usr/bin/env python3
"""Canonical TF++ agent with a post-PID, turn-gated vehicle steer adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tfpp_sensor_rig_agent import SensorRigAgent  # noqa: E402


def get_entry_point() -> str:
    return "VehicleSteerAdapterSensorRigAgent"


class VehicleSteerAdapterSensorRigAgent(SensorRigAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)
        self._load_vehicle_steer_adapter()

    def _load_vehicle_steer_adapter(self) -> None:
        checkpoint_path = os.environ.get("TFPP_VEHICLE_STEER_ADAPTER_CHECKPOINT", "")
        if not checkpoint_path:
            raise ValueError("TFPP_VEHICLE_STEER_ADAPTER_CHECKPOINT is required")
        adapter_root = Path(os.environ.get("ADAPTER_ROOT", Path(__file__).resolve().parents[1])).expanduser()
        if str(adapter_root) not in sys.path:
            sys.path.insert(0, str(adapter_root))
        from teach2drive_adapter.train_tfpp_vehicle_steer_adapter import TurnGatedSteerResidualAdapter

        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=self.device)
        metadata = checkpoint.get("metadata", {}).get("vehicle_steer_adapter", {})
        self._vehicle_steer_adapter = TurnGatedSteerResidualAdapter(
            checkpoint_dim=int(metadata.get("checkpoint_dim", 20)),
            hidden_dim=int(metadata.get("hidden_dim", 128)),
            max_delta=float(metadata.get("max_delta", 0.20)),
            turn_threshold=float(metadata.get("turn_threshold", 0.035)),
            full_turn_threshold=float(metadata.get("full_turn_threshold", 0.12)),
            dropout=0.0,
            use_yaw_rate=bool(metadata.get("use_yaw_rate", False)),
            adapter_mode=str(metadata.get("adapter_mode", "residual")),
            minimum_gain=float(metadata.get("minimum_gain", 0.10)),
            maximum_gain=float(metadata.get("maximum_gain", 1.20)),
        ).to(self.device)
        missing, unexpected = self._vehicle_steer_adapter.load_state_dict(
            checkpoint["vehicle_steer_adapter_state"], strict=False
        )
        self._vehicle_steer_adapter.eval()
        self._vehicle_steer_blend = float(os.environ.get("TFPP_VEHICLE_STEER_ADAPTER_BLEND", "1.0") or "1.0")
        self._vehicle_yaw_rate = 0.0
        self._patch_lateral_controllers()
        print(
            "[VehicleSteerAdapterSensorRigAgent] loaded "
            f"checkpoint={Path(checkpoint_path).expanduser()} blend={self._vehicle_steer_blend:.3f} "
            f"max_delta={float(metadata.get('max_delta', 0.20)):.3f} "
            f"use_yaw_rate={int(bool(metadata.get('use_yaw_rate', False)))} "
            f"mode={metadata.get('adapter_mode', 'residual')} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    def _patch_lateral_controllers(self) -> None:
        for net in self.nets:
            original = net.control_pid_direct

            def adapted_control(pred_checkpoints, pred_target_speed, speed, *args, _original=original, _net=net, **kwargs):
                steer, throttle, brake = _original(pred_checkpoints, pred_target_speed, speed, *args, **kwargs)
                controller = _net.lateral_pid_controller
                history = list(getattr(controller, "error_history", []))
                pid_error = float(history[-1]) if history else 0.0
                pid_derivative = float(history[-1] - history[-2]) if len(history) >= 2 else 0.0
                pid_integral = float(np.mean(history)) if history else 0.0
                checkpoint = torch.as_tensor(pred_checkpoints, dtype=torch.float32, device=self.device).reshape(1, -1)
                if torch.is_tensor(speed):
                    speed_value = float(speed.detach().reshape(-1)[0].item())
                else:
                    speed_value = float(np.asarray(speed).reshape(-1)[0])
                with torch.no_grad():
                    output = self._vehicle_steer_adapter(
                        checkpoint,
                        torch.tensor([speed_value], dtype=torch.float32, device=self.device),
                        torch.tensor([float(pred_target_speed)], dtype=torch.float32, device=self.device),
                        torch.tensor([float(steer)], dtype=torch.float32, device=self.device),
                        torch.tensor([pid_error], dtype=torch.float32, device=self.device),
                        torch.tensor([pid_derivative], dtype=torch.float32, device=self.device),
                        torch.tensor([pid_integral], dtype=torch.float32, device=self.device),
                        torch.tensor([self._vehicle_yaw_rate], dtype=torch.float32, device=self.device),
                    )
                delta = float(output["delta"].item()) * self._vehicle_steer_blend
                return float(np.clip(float(steer) + delta, -1.0, 1.0)), throttle, brake

            net.control_pid_direct = adapted_control

    def run_step(self, input_data, timestamp, sensors=None):
        imu_packet = input_data.get("imu") if isinstance(input_data, dict) else None
        if imu_packet is not None:
            imu = np.asarray(imu_packet[1], dtype=np.float32).reshape(-1)
            # CARLA IMU layout: accel xyz, gyro xyz, compass. Gyro z is yaw rate [rad/s].
            if imu.size >= 7 and np.isfinite(imu[5]):
                self._vehicle_yaw_rate = float(imu[5])
        return super().run_step(input_data, timestamp, sensors=sensors)
