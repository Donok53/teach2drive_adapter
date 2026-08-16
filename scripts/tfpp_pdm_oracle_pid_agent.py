#!/usr/bin/env python3
"""Diagnostic agent: PDM-Lite oracle outputs through the original TF++ PID.

This agent deliberately uses privileged simulator state.  It is not a deployable
policy; it isolates whether Tesla failures remain when checkpoint and target
speed prediction errors are removed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import carla
import jsonpickle
import numpy as np

from leaderboard.autoagents import autonomous_agent
from autopilot import AutoPilot
from config import GlobalConfig
from data import CARLA_Data
from nav_planner import LateralPIDController, get_throttle


def get_entry_point() -> str:
    return "PDMOracleTFPPPIDAgent"


class PDMOracleTFPPPIDAgent(AutoPilot):
    """Run PDM-Lite planning/hazard logic but replace its controller with TF++."""

    def __init__(self, host="localhost", port=2000, debug=0):
        # The non-local leaderboard evaluator constructs agents as
        # (host, port, debug), while AutoPilot derives from the local agent base
        # whose constructor expects (config_path, route_index).  Neither value is
        # consumed before setup(), so bridge the two constructor conventions.
        super().__init__(str(host), route_index=None)
        self._oracle_host = str(host)
        self._oracle_port = int(port)
        self._oracle_debug = int(debug)

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)

        # CARLA Garage's expert was developed against its autopilot scenario
        # runner fork.  The normal evaluation runner lacks this optional list;
        # an empty list is the correct state for missions without the special
        # route-obstacle bookkeeping used by that fork.
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        if not hasattr(CarlaDataProvider, "active_scenarios"):
            CarlaDataProvider.active_scenarios = []

        # The standard mission runner evaluates the SENSORS track.  The oracle
        # reads the world through CarlaDataProvider, so no OpenDRIVE sensor is
        # needed even though the diagnostic itself is privileged.
        self.track = autonomous_agent.Track.SENSORS

        config_path = Path(path_to_conf_file).expanduser() / "config.json"
        loaded = jsonpickle.decode(config_path.read_text(encoding="utf-8"))
        self._tfpp_config = GlobalConfig()
        if isinstance(loaded, dict):
            self._tfpp_config.__dict__.update(loaded)
        else:
            self._tfpp_config.__dict__.update(loaded.__dict__)

        # Keep AutoPilot's own config/controllers untouched.  The pretrained
        # checkpoint config is used only by the TF++ output formatting and PID.
        self._tfpp_lateral_pid = LateralPIDController(self._tfpp_config)
        self._tfpp_data = CARLA_Data(root=[], config=self._tfpp_config, shared_dict=None)
        self._oracle_step = 0
        self._trace_handle = None
        trace_path = os.environ.get("TFPP_ORACLE_TRACE_PATH", "")
        if trace_path:
            trace = Path(trace_path).expanduser()
            trace.parent.mkdir(parents=True, exist_ok=True)
            self._trace_handle = trace.open("a", encoding="utf-8", buffering=1)

        print(
            "[PDMOracleTFPPPID] privileged PDM route+target_speed -> original TF++ PID "
            f"config={config_path} trace={trace_path or 'off'}",
            flush=True,
        )

    def sensors(self):
        # AutoPilot obtains the map/world from CarlaDataProvider in _init().
        return [
            {
                "type": "sensor.other.imu",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "imu",
            },
            {
                "type": "sensor.speedometer",
                "reading_frequency": 20,
                "id": "speed",
            },
        ]

    def _tfpp_control(self, driving_data):
        route = np.asarray(driving_data["route"], dtype=np.float32).reshape(-1, 2)
        if route.shape[0] == 0:
            return 0.0, 0.0, True, route

        route_count = int(getattr(self._tfpp_config, "num_route_points", 20))
        if route.shape[0] < route_count:
            route = np.vstack((route, np.tile(route[-1], (route_count - route.shape[0], 1))))
        else:
            route = route[:route_count]
        if bool(getattr(self._tfpp_config, "smooth_route", 1)):
            route = self._tfpp_data.smooth_path(route)

        checkpoint_count = int(getattr(self._tfpp_config, "predict_checkpoint_len", 10))
        checkpoints = np.asarray(route[:checkpoint_count], dtype=np.float32)
        speed = float(driving_data["speed"])
        target_speed = max(float(driving_data["target_speed"]), 0.0)
        brake = target_speed < 0.01 or speed / max(target_speed, 1e-6) > float(self._tfpp_config.brake_ratio)
        steer = round(float(np.clip(self._tfpp_lateral_pid.step(checkpoints, speed, 0, 0), -1.0, 1.0)), 3)
        throttle, control_brake = get_throttle(self._tfpp_config, brake, target_speed, speed)
        return (
            steer,
            float(np.clip(throttle, 0.0, float(self._tfpp_config.clip_throttle))),
            bool(control_brake),
            checkpoints,
        )

    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        self.step += 1
        if not self.initialized:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

            self._init(CarlaDataProvider.get_map())

        # False is the normal PDM-Lite expert mode.  Passing True selects the
        # PlanT path and deliberately disables simulator-actor forecasting,
        # which would make this collision/hazard oracle invalid.
        expert_control, driving_data = self._get_control(input_data, plant=False)
        steer, throttle, brake, checkpoints = self._tfpp_control(driving_data)

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        if self._trace_handle is not None:
            record = {
                "step": int(self._oracle_step),
                "timestamp": float(timestamp),
                "speed": float(driving_data["speed"]),
                "oracle_target_speed": float(driving_data["target_speed"]),
                "oracle_checkpoints": checkpoints.tolist(),
                "tfpp_steer": steer,
                "tfpp_throttle": throttle,
                "tfpp_brake": bool(brake),
                "expert_steer": float(driving_data["steer"]),
                "expert_throttle": float(driving_data["throttle"]),
                "expert_brake": bool(expert_control.brake),
                "junction": bool(driving_data["junction"]),
                "vehicle_hazard": bool(driving_data["vehicle_hazard"]),
                "light_hazard": bool(driving_data["light_hazard"]),
                "walker_hazard": bool(driving_data["walker_hazard"]),
                "stop_sign_hazard": bool(driving_data["stop_sign_hazard"]),
            }
            self._trace_handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._oracle_step += 1
        return control

    def destroy(self, results=None):
        try:
            if self._trace_handle is not None:
                self._trace_handle.close()
                self._trace_handle = None
        finally:
            return super().destroy(results=results)
