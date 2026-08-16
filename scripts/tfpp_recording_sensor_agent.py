#!/usr/bin/env python3
"""CARLA Garage TransFuser++ agent wrapper that records the model input camera.

This records frames directly from the leaderboard input_data dictionary, so it
does not need an extra CARLA camera actor or a second recording client.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2

from sensor_agent import SensorAgent


def get_entry_point() -> str:
    return "RecordingSensorAgent"


def _truthy(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


class RecordingSensorAgent(SensorAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)
        self._record_enabled = _truthy(os.environ.get("TFPP_AGENT_RECORD_VIDEO"), default=True)
        self._record_sensor_id = os.environ.get("TFPP_RECORD_SENSOR_ID", "rgb_front")
        self._record_fps = float(os.environ.get("TFPP_RECORD_FPS", "20"))
        self._record_output = os.environ.get("TFPP_RECORD_OUTPUT") or os.environ.get("VIDEO_OUTPUT", "")
        self._record_frame_dir = os.environ.get("TFPP_RECORD_FRAME_DIR", "")
        self._record_image_format = os.environ.get("TFPP_RECORD_IMAGE_FORMAT", "jpg").lower().lstrip(".")
        self._record_jpeg_quality = int(os.environ.get("TFPP_RECORD_JPEG_QUALITY", "98"))
        self._record_every_n = max(1, int(os.environ.get("TFPP_RECORD_EVERY_N", "1")))
        self._record_scale = float(os.environ.get("TFPP_RECORD_SCALE", "1.0"))
        self._record_draw_step = _truthy(os.environ.get("TFPP_RECORD_DRAW_STEP"), default=True)
        self._record_writer = None
        self._record_frames = 0
        if self._record_enabled and self._record_output:
            Path(self._record_output).expanduser().parent.mkdir(parents=True, exist_ok=True)
        if self._record_enabled and self._record_frame_dir:
            Path(self._record_frame_dir).expanduser().mkdir(parents=True, exist_ok=True)
        if self._record_enabled:
            print(
                "[RecordingSensorAgent] recording "
                f"sensor={self._record_sensor_id} fps={self._record_fps} output={self._record_output} "
                f"frame_dir={self._record_frame_dir} every_n={self._record_every_n} scale={self._record_scale}",
                flush=True,
            )

    def _write_record_frame(self, input_data: dict[str, Any]) -> None:
        if not self._record_enabled or (not self._record_output and not self._record_frame_dir):
            return
        step = int(getattr(self, "step", -1))
        if step >= 0 and step % self._record_every_n != 0:
            return
        if self._record_sensor_id not in input_data:
            if self._record_frames == 0:
                print(f"[RecordingSensorAgent] missing sensor {self._record_sensor_id!r}", flush=True)
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
        if self._record_writer is None and self._record_output:
            self._record_writer = cv2.VideoWriter(
                str(Path(self._record_output).expanduser()),
                cv2.VideoWriter_fourcc(*os.environ.get("TFPP_RECORD_CODEC", "mp4v")),
                self._record_fps,
                (width, height),
            )
            if not self._record_writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {self._record_output}")
        if self._record_draw_step:
            cv2.putText(
                frame,
                f"step={step}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if self._record_writer is not None:
            self._record_writer.write(frame)
        if self._record_frame_dir:
            frame_dir = Path(self._record_frame_dir).expanduser()
            suffix = "png" if self._record_image_format == "png" else "jpg"
            path = frame_dir / f"{self._record_frames:06d}.{suffix}"
            if suffix == "png":
                cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 2])
            else:
                cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._record_jpeg_quality])
        self._record_frames += 1

    def run_step(self, input_data, timestamp, sensors=None):
        self._write_record_frame(input_data)
        return super().run_step(input_data, timestamp, sensors=sensors)

    def destroy(self, results=None):
        try:
            if self._record_writer is not None:
                self._record_writer.release()
                print(f"[RecordingSensorAgent] saved_frames={self._record_frames} output={self._record_output}", flush=True)
        finally:
            self._record_writer = None
            return super().destroy(results=results)
