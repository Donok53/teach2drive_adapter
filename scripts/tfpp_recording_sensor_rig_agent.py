#!/usr/bin/env python3
"""TransFuser++ leaderboard agent with alternate sensor rig and input recording."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import cv2

from sensor_agent import SensorAgent


def get_entry_point() -> str:
    return "RecordingSensorRigAgent"


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


def _truthy(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


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


class RecordingSensorRigAgent(SensorAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)
        self._apply_sensor_rig()
        self._setup_recording()

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
            "[RecordingSensorRigAgent] applied rig="
            f"{source} camera={camera_name} pos={self.config.camera_pos} rot={self.config.camera_rot_0} "
            f"lidar={lidar_name} pos={self.config.lidar_pos} rot={self.config.lidar_rot}",
            flush=True,
        )

    def _setup_recording(self) -> None:
        self._record_enabled = _truthy(os.environ.get("TFPP_AGENT_RECORD_VIDEO"), default=True)
        self._record_sensor_id = os.environ.get("TFPP_RECORD_SENSOR_ID", "rgb_front")
        self._record_fps = _env_float("TFPP_RECORD_FPS", 20.0)
        self._record_output = os.environ.get("TFPP_RECORD_OUTPUT") or os.environ.get("VIDEO_OUTPUT", "")
        self._record_every_n = max(1, _env_int("TFPP_RECORD_EVERY_N", 1))
        self._record_scale = _env_float("TFPP_RECORD_SCALE", 1.0)
        self._record_draw_step = _truthy(os.environ.get("TFPP_RECORD_DRAW_STEP"), default=True)
        self._record_writer = None
        self._record_frames = 0
        if self._record_enabled and self._record_output:
            Path(self._record_output).expanduser().parent.mkdir(parents=True, exist_ok=True)
            print(
                "[RecordingSensorRigAgent] recording "
                f"sensor={self._record_sensor_id} fps={self._record_fps} output={self._record_output}",
                flush=True,
            )

    def _write_record_frame(self, input_data: dict[str, Any]) -> None:
        if not self._record_enabled or not self._record_output:
            return
        step = int(getattr(self, "step", -1))
        if step >= 0 and step % self._record_every_n != 0:
            return
        if self._record_sensor_id not in input_data:
            if self._record_frames == 0:
                print(f"[RecordingSensorRigAgent] missing sensor {self._record_sensor_id!r}", flush=True)
            return
        frame = input_data[self._record_sensor_id][1][:, :, :3].copy()
        if self._record_scale > 0 and abs(self._record_scale - 1.0) > 1e-6:
            frame = cv2.resize(
                frame,
                None,
                fx=self._record_scale,
                fy=self._record_scale,
                interpolation=cv2.INTER_LINEAR if self._record_scale > 1.0 else cv2.INTER_AREA,
            )
        height, width = frame.shape[:2]
        if self._record_writer is None:
            self._record_writer = cv2.VideoWriter(
                str(Path(self._record_output).expanduser()),
                cv2.VideoWriter_fourcc(*os.environ.get("TFPP_RECORD_CODEC", "mp4v")),
                self._record_fps,
                (width, height),
            )
            if not self._record_writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {self._record_output}")
        if self._record_draw_step:
            cv2.putText(frame, f"step={step}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        self._record_writer.write(frame)
        self._record_frames += 1

    def run_step(self, input_data, timestamp, sensors=None):
        self._write_record_frame(input_data)
        return super().run_step(input_data, timestamp, sensors=sensors)

    def destroy(self, results=None):
        try:
            if self._record_writer is not None:
                self._record_writer.release()
                print(f"[RecordingSensorRigAgent] saved_frames={self._record_frames} output={self._record_output}", flush=True)
        finally:
            self._record_writer = None
            return super().destroy(results=results)
