#!/usr/bin/env python3
"""TransFuser++ Leaderboard agent with an alternate sensor rig.

This wrapper keeps the official CARLA Garage TransFuser++ model and control
stack unchanged, but mounts the input RGB camera and LiDAR at positions that
match the Teach2Drive collection rigs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from sensor_agent import SensorAgent


def get_entry_point() -> str:
  return "SensorRigAgent"


_BUILTIN_RIGS: dict[str, dict[str, Any]] = {
    "tfpp_ego": {
        "cameras": {
            "front": {"x": -1.5, "y": 0.0, "z": 2.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        },
        "lidars": {
            "top": {"x": 0.0, "y": 0.0, "z": 2.5, "roll": 0.0, "pitch": 0.0, "yaw": -90.0},
        },
    },
    "front_triplet_shifted": {
        "cameras": {
            "left": {"x": 1.20, "y": -0.38, "z": 1.95, "roll": 0.0, "pitch": 0.0, "yaw": -12.0},
            "front": {"x": 1.25, "y": 0.0, "z": 1.95, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "right": {"x": 1.20, "y": 0.38, "z": 1.95, "roll": 0.0, "pitch": 0.0, "yaw": 12.0},
        },
        "lidars": {
            "top": {"x": 0.20, "y": 0.0, "z": 2.35, "roll": 0.0, "pitch": 0.0, "yaw": -90.0},
        },
    },
}


def _pose_vec(pose: Mapping[str, Any], keys: tuple[str, str, str]) -> list[float]:
  return [float(pose[key]) for key in keys]


def _load_layout() -> tuple[str, Mapping[str, Any]]:
  layout_path = os.environ.get("TFPP_SENSOR_RIG_JSON")
  if layout_path:
    path = Path(layout_path).expanduser()
    return str(path), json.loads(path.read_text(encoding="utf-8"))

  rig = os.environ.get("TFPP_SENSOR_RIG", "front_triplet_shifted")
  if rig not in _BUILTIN_RIGS:
    choices = ", ".join(sorted(_BUILTIN_RIGS))
    raise ValueError(f"Unknown TFPP_SENSOR_RIG={rig!r}. Available: {choices}")
  return rig, _BUILTIN_RIGS[rig]


class SensorRigAgent(SensorAgent):
  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)
    self._apply_sensor_rig()

  def _apply_sensor_rig(self) -> None:
    source, layout = _load_layout()
    camera_name = os.environ.get("TFPP_SENSOR_CAMERA", "front")
    lidar_name = os.environ.get("TFPP_SENSOR_LIDAR", "top")

    cameras = layout.get("cameras", {})
    lidars = layout.get("lidars", layout.get("lidar", {}))
    if camera_name not in cameras:
      raise ValueError(f"Camera {camera_name!r} not found in rig {source!r}: {sorted(cameras)}")
    if lidar_name not in lidars:
      raise ValueError(f"LiDAR {lidar_name!r} not found in rig {source!r}: {sorted(lidars)}")

    camera = cameras[camera_name]
    lidar = lidars[lidar_name]
    self.config.camera_pos = _pose_vec(camera, ("x", "y", "z"))
    self.config.camera_rot_0 = _pose_vec(camera, ("roll", "pitch", "yaw"))
    self.config.lidar_pos = _pose_vec(lidar, ("x", "y", "z"))
    self.config.lidar_rot = _pose_vec(lidar, ("roll", "pitch", "yaw"))

    print(
        "[SensorRigAgent] applied rig="
        f"{source} camera={camera_name} pos={self.config.camera_pos} rot={self.config.camera_rot_0} "
        f"lidar={lidar_name} pos={self.config.lidar_pos} rot={self.config.lidar_rot}"
    )
