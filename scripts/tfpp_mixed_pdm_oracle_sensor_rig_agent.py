#!/usr/bin/env python3
"""TF++ ablation agent mixing one model output with one PDM-Lite oracle output.

Modes:
  checkpoint: PDM route checkpoints + TF++ predicted target speed
  speed:      TF++ predicted checkpoints + PDM target speed

This is a privileged diagnostic agent, not a deployable policy.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tfpp_sensor_rig_agent import SensorRigAgent  # noqa: E402


def get_entry_point() -> str:
    return "MixedPDMOracleSensorRigAgent"


class MixedPDMOracleSensorRigAgent(SensorRigAgent):
    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        # The normal SensorAgent base discards the dense route while PDM-Lite's
        # local-agent base retains it.  Preserve it before the normal downsample.
        self.org_dense_route_gps = global_plan_gps
        self.org_dense_route_world_coord = global_plan_world_coord
        return super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)

        mode = os.environ.get("TFPP_ORACLE_MIX_MODE", "").strip().lower()
        if mode not in {"checkpoint", "speed"}:
            raise ValueError("TFPP_ORACLE_MIX_MODE must be 'checkpoint' or 'speed'")
        self._oracle_mix_mode = mode
        self._oracle_checkpoints = None
        self._oracle_target_speed = None
        self._oracle_driving_data = None
        self._oracle_trace_handle = None
        self._oracle_trace_step = 0

        from autopilot import AutoPilot
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        if not hasattr(CarlaDataProvider, "active_scenarios"):
            CarlaDataProvider.active_scenarios = []

        # A shadow PDM-Lite expert reads the same live ego/world state and only
        # supplies privileged planner outputs.  Its control is never applied.
        self._pdm_oracle = AutoPilot("", route_index=None)
        self._pdm_oracle.org_dense_route_gps = self.org_dense_route_gps
        self._pdm_oracle.org_dense_route_world_coord = self.org_dense_route_world_coord
        self._pdm_oracle._global_plan = self._global_plan
        self._pdm_oracle._global_plan_world_coord = self._global_plan_world_coord
        self._pdm_oracle.setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)

        trace_path = os.environ.get("TFPP_ORACLE_TRACE_PATH", "")
        if trace_path:
            path = Path(trace_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._oracle_trace_handle = path.open("a", encoding="utf-8", buffering=1)

        self._patch_controller()
        print(
            f"[MixedPDMOracle] mode={mode} "
            f"trace={trace_path or 'off'} checkpoint_oracle={mode == 'checkpoint'} "
            f"speed_oracle={mode == 'speed'}",
            flush=True,
        )

    def _format_oracle_checkpoints(self, route) -> np.ndarray:
        route = np.asarray(route, dtype=np.float32).reshape(-1, 2)
        route_count = int(getattr(self.config, "num_route_points", 20))
        if route.shape[0] == 0:
            return np.zeros((int(getattr(self.config, "predict_checkpoint_len", 10)), 2), dtype=np.float32)
        if route.shape[0] < route_count:
            route = np.vstack((route, np.tile(route[-1], (route_count - route.shape[0], 1))))
        else:
            route = route[:route_count]
        if bool(getattr(self.config, "smooth_route", 1)):
            route = self.data.smooth_path(route)
        count = int(getattr(self.config, "predict_checkpoint_len", 10))
        return np.asarray(route[:count], dtype=np.float32)

    def _patch_controller(self) -> None:
        original = self.nets[0].control_pid_direct

        def mixed_control(pred_checkpoints, pred_target_speed, speed, *args, **kwargs):
            if self._oracle_checkpoints is None or self._oracle_target_speed is None:
                return original(pred_checkpoints, pred_target_speed, speed, *args, **kwargs)

            model_checkpoints = np.asarray(pred_checkpoints, dtype=np.float32)
            model_target_speed = float(pred_target_speed)
            if self._oracle_mix_mode == "checkpoint":
                used_checkpoints = self._oracle_checkpoints
                used_target_speed = model_target_speed
            else:
                used_checkpoints = model_checkpoints
                used_target_speed = float(self._oracle_target_speed)

            steer, throttle, brake = original(used_checkpoints, used_target_speed, speed, *args, **kwargs)
            if self._oracle_trace_handle is not None:
                record = {
                    "step": int(self._oracle_trace_step),
                    "mode": self._oracle_mix_mode,
                    "model_checkpoints": model_checkpoints.tolist(),
                    "oracle_checkpoints": self._oracle_checkpoints.tolist(),
                    "model_target_speed": model_target_speed,
                    "oracle_target_speed": float(self._oracle_target_speed),
                    "used_target_speed": float(used_target_speed),
                    "steer": float(steer),
                    "throttle": float(throttle),
                    "brake": bool(brake),
                }
                if self._oracle_driving_data is not None:
                    record.update(
                        {
                            "expert_steer": float(self._oracle_driving_data["steer"]),
                            "expert_throttle": float(self._oracle_driving_data["throttle"]),
                            "junction": bool(self._oracle_driving_data["junction"]),
                            "vehicle_hazard": bool(self._oracle_driving_data["vehicle_hazard"]),
                            "light_hazard": bool(self._oracle_driving_data["light_hazard"]),
                            "walker_hazard": bool(self._oracle_driving_data["walker_hazard"]),
                            "stop_sign_hazard": bool(self._oracle_driving_data["stop_sign_hazard"]),
                        }
                    )
                self._oracle_trace_handle.write(json.dumps(record, sort_keys=True) + "\n")
                self._oracle_trace_step += 1
            return steer, throttle, brake

        self.nets[0].control_pid_direct = mixed_control

    def run_step(self, input_data, timestamp, sensors=None):
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        self._pdm_oracle.step += 1
        if not self._pdm_oracle.initialized:
            self._pdm_oracle._init(CarlaDataProvider.get_map())
        _expert_control, driving_data = self._pdm_oracle._get_control(input_data, plant=False)
        self._oracle_driving_data = driving_data
        self._oracle_checkpoints = self._format_oracle_checkpoints(driving_data["route"])
        self._oracle_target_speed = max(float(driving_data["target_speed"]), 0.0)
        return super().run_step(input_data, timestamp, sensors=sensors)

    def destroy(self, results=None):
        try:
            if self._oracle_trace_handle is not None:
                self._oracle_trace_handle.close()
                self._oracle_trace_handle = None
            if getattr(self, "_pdm_oracle", None) is not None:
                self._pdm_oracle.destroy(results=results)
                self._pdm_oracle = None
        finally:
            return super().destroy(results=results)
