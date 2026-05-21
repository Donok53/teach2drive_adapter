#!/usr/bin/env python3
"""TransFuser++ leaderboard agent with stage feature adapters and fused-feature adapter."""

from __future__ import annotations

import atexit
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import torch
import torch.nn.functional as F

from sensor_agent import SensorAgent


def get_entry_point() -> str:
    return "FeatureThenFusionAdapterSensorRigAgent"


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


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    return int(value)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return bool(default)
    return value.lower() in {"1", "true", "yes", "y", "on"}


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


def _ensure_adapter_import_path() -> None:
    root = Path(os.environ.get("ADAPTER_ROOT", "/home/byeongjae/code/teach2drive_adapter")).expanduser()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class FeatureThenFusionAdapterSensorRigAgent(SensorAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)
        self._rig_source, self._rig_layout = _load_layout()
        self._apply_sensor_rig()
        self._load_adapter()
        self._patch_backbones()
        self._setup_recording()

    def _apply_sensor_rig(self) -> None:
        camera_name = os.environ.get("TFPP_SENSOR_CAMERA", "front")
        lidar_name = os.environ.get("TFPP_SENSOR_LIDAR", "top")

        cameras = self._rig_layout.get("cameras", {})
        lidars = self._rig_layout.get("lidars", self._rig_layout.get("lidar", {}))
        if camera_name not in cameras:
            raise ValueError(f"Camera {camera_name!r} not found in rig {self._rig_source!r}: {sorted(cameras)}")
        if lidar_name not in lidars:
            raise ValueError(f"LiDAR {lidar_name!r} not found in rig {self._rig_source!r}: {sorted(lidars)}")

        camera = cameras[camera_name]
        lidar = lidars[lidar_name]
        self.config.camera_pos = _pose_vec(camera, ("x", "y", "z"))
        self.config.camera_rot_0 = _pose_vec(camera, ("roll", "pitch", "yaw"))
        self.config.lidar_pos = _pose_vec(lidar, ("x", "y", "z"))
        self.config.lidar_rot = _pose_vec(lidar, ("roll", "pitch", "yaw"))
        print(
            "[FeatureThenFusionAdapterSensorRigAgent] applied rig="
            f"{self._rig_source} camera={camera_name} pos={self.config.camera_pos} rot={self.config.camera_rot_0} "
            f"lidar={lidar_name} pos={self.config.lidar_pos} rot={self.config.lidar_rot}",
            flush=True,
        )

    def _load_adapter(self) -> None:
        checkpoint_path = (
            os.environ.get("TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT")
            or os.environ.get("TFPP_ADAPTER_CHECKPOINT")
            or ""
        )
        if not checkpoint_path:
            raise ValueError("TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT or TFPP_ADAPTER_CHECKPOINT is required")
        _ensure_adapter_import_path()
        from teach2drive_adapter.train_transfuserpp_feature_then_fusion_adapter import (
            ExtrinsicAwareFeatureThenFusionAdapter,
            FeatureThenFusionAdapter,
            build_extrinsic_vector,
        )

        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=self.device)
        metadata = checkpoint.get("metadata", {})
        args = metadata.get("args", {})
        stage_shapes_raw = checkpoint.get("stage_feature_shapes", metadata.get("stage_feature_shapes", {}))
        fused_shape_raw = checkpoint.get("fused_feature_shape", metadata.get("fused_feature_shape", []))
        self._stage_feature_shapes = {str(name): tuple(int(v) for v in shape) for name, shape in stage_shapes_raw.items()}
        self._fused_feature_shape = tuple(int(v) for v in fused_shape_raw)
        if not self._stage_feature_shapes or len(self._fused_feature_shape) != 3:
            raise ValueError("Invalid feature-then-fusion adapter shapes in checkpoint")
        self._extrinsic_aware = bool(metadata.get("extrinsic_aware", False)) or metadata.get("mode") == "transfuserpp_extrinsic_feature_then_fusion_adapter"
        if self._extrinsic_aware:
            extrinsic_vector = metadata.get("extrinsic_vector")
            if not extrinsic_vector:
                extrinsic_vector = build_extrinsic_vector(str(metadata.get("source_profile", "front_triplet_shifted")))
            self._adapter = ExtrinsicAwareFeatureThenFusionAdapter(
                stage_feature_shapes=self._stage_feature_shapes,
                fused_feature_shape=self._fused_feature_shape,
                extrinsic_vector=extrinsic_vector,
                hidden_channels=int(args.get("hidden_channels", 0)),
                blocks=int(args.get("blocks", 2)),
                dropout=float(args.get("dropout", 0.0)),
                extrinsic_hidden_dim=int(args.get("extrinsic_hidden_dim", 64)),
                extrinsic_dropout=float(args.get("extrinsic_dropout", 0.0)),
            ).to(self.device)
        else:
            self._adapter = FeatureThenFusionAdapter(
                stage_feature_shapes=self._stage_feature_shapes,
                fused_feature_shape=self._fused_feature_shape,
                hidden_channels=int(args.get("hidden_channels", 0)),
                blocks=int(args.get("blocks", 2)),
                dropout=float(args.get("dropout", 0.0)),
            ).to(self.device)
        missing, unexpected = self._adapter.load_state_dict(checkpoint["model_state"], strict=False)
        self._adapter.eval()
        shared_blend = _env_float("TFPP_FEATURE_ADAPTER_BLEND", 1.0)
        self._stage_blend = _env_float("TFPP_STAGE_FEATURE_ADAPTER_BLEND", shared_blend)
        self._fusion_blend = _env_float("TFPP_FUSION_ADAPTER_BLEND", shared_blend)
        print(
            "[FeatureThenFusionAdapterSensorRigAgent] loaded adapter "
            f"checkpoint={Path(checkpoint_path).expanduser()} "
            f"extrinsic_aware={self._extrinsic_aware} "
            f"stage_blend={self._stage_blend:.3f} fusion_blend={self._fusion_blend:.3f} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    def _adapt_stage_pair(self, layer_idx: int, image_embd_layer: torch.Tensor, lidar_embd_layer: torch.Tensor):
        image_name = f"layer_{int(layer_idx)}_image"
        lidar_name = f"layer_{int(layer_idx)}_lidar"
        expected_image = self._stage_feature_shapes.get(image_name)
        expected_lidar = self._stage_feature_shapes.get(lidar_name)
        got_image = tuple(int(v) for v in image_embd_layer.shape[1:])
        got_lidar = tuple(int(v) for v in lidar_embd_layer.shape[1:])
        if got_image != expected_image or got_lidar != expected_lidar:
            key = f"{image_name}:{got_image}:{got_lidar}"
            warned = getattr(self, "_stage_shape_warnings", set())
            if key not in warned:
                warned.add(key)
                self._stage_shape_warnings = warned
                print(
                    "[FeatureThenFusionAdapterSensorRigAgent] stage shape mismatch "
                    f"layer={layer_idx} image={got_image}/{expected_image} lidar={got_lidar}/{expected_lidar}; skipping stage adapter",
                    flush=True,
                )
            return image_embd_layer, lidar_embd_layer
        adapted_image, adapted_lidar = self._adapter.adapt_layer(
            int(layer_idx),
            image_embd_layer.float(),
            lidar_embd_layer.float(),
        )
        adapted_image = adapted_image.to(dtype=image_embd_layer.dtype)
        adapted_lidar = adapted_lidar.to(dtype=lidar_embd_layer.dtype)
        blend = float(self._stage_blend)
        if blend < 1.0:
            adapted_image = image_embd_layer + blend * (adapted_image - image_embd_layer)
            adapted_lidar = lidar_embd_layer + blend * (adapted_lidar - lidar_embd_layer)
        return adapted_image, adapted_lidar

    def _adapt_fused(self, fused: torch.Tensor, net_index: int) -> torch.Tensor:
        if fused.ndim != 4:
            return fused
        got = tuple(int(v) for v in fused.shape[1:])
        if got != self._fused_feature_shape:
            if getattr(self, "_fused_shape_warned", False) is False:
                print(
                    "[FeatureThenFusionAdapterSensorRigAgent] fused feature shape mismatch "
                    f"net={net_index} got={got} expected={self._fused_feature_shape}; skipping fusion adapter",
                    flush=True,
                )
                self._fused_shape_warned = True
            return fused
        adapted = self._adapter.adapt_fused(fused.float()).to(dtype=fused.dtype)
        blend = float(self._fusion_blend)
        if blend < 1.0:
            adapted = fused + blend * (adapted - fused)
        return adapted

    def _patch_backbones(self) -> None:
        for index, net in enumerate(self.nets):
            backbone = net.backbone
            original_forward = backbone.forward

            def adapted_fuse_features(image_features, lidar_features, layer_idx, _backbone=backbone):
                idx = int(layer_idx)
                image_embd_layer = _backbone.avgpool_img(image_features)
                lidar_embd_layer = _backbone.avgpool_lidar(lidar_features)
                lidar_embd_layer = _backbone.lidar_channel_to_img[idx](lidar_embd_layer)

                image_embd_layer, lidar_embd_layer = self._adapt_stage_pair(idx, image_embd_layer, lidar_embd_layer)
                image_features_layer, lidar_features_layer = _backbone.transformers[idx](image_embd_layer, lidar_embd_layer)
                lidar_features_layer = _backbone.img_channel_to_lidar[idx](lidar_features_layer)

                image_features_layer = F.interpolate(
                    image_features_layer,
                    size=(image_features.shape[2], image_features.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                if _backbone.lidar_video:
                    lidar_features_layer = F.interpolate(
                        lidar_features_layer,
                        size=(lidar_features.shape[2], lidar_features.shape[3], lidar_features.shape[4]),
                        mode="trilinear",
                        align_corners=False,
                    )
                else:
                    lidar_features_layer = F.interpolate(
                        lidar_features_layer,
                        size=(lidar_features.shape[2], lidar_features.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                return image_features + image_features_layer, lidar_features + lidar_features_layer

            backbone.fuse_features = adapted_fuse_features

            def adapted_forward(*args, _original_forward=original_forward, _index=index, **kwargs):
                output = _original_forward(*args, **kwargs)
                if not isinstance(output, (tuple, list)) or len(output) < 2:
                    return output
                adapted = self._adapt_fused(output[1], _index)
                if isinstance(output, tuple):
                    return (output[0], adapted, *output[2:])
                out = list(output)
                out[1] = adapted
                return out

            backbone.forward = adapted_forward
        print(f"[FeatureThenFusionAdapterSensorRigAgent] patched {len(self.nets)} backbone(s)", flush=True)

    def _setup_recording(self) -> None:
        self._record_enabled = _truthy(os.environ.get("TFPP_AGENT_RECORD_VIDEO"), default=False)
        self._record_sensor_id = os.environ.get("TFPP_RECORD_SENSOR_ID", "rgb_front")
        self._record_fps = _env_float("TFPP_RECORD_FPS", 20.0)
        self._record_output = os.environ.get("TFPP_RECORD_OUTPUT") or os.environ.get("VIDEO_OUTPUT", "")
        self._record_every_n = max(1, _env_int("TFPP_RECORD_EVERY_N", 1))
        self._record_scale = _env_float("TFPP_RECORD_SCALE", 1.0)
        self._record_writer = None
        self._record_frames = 0
        if self._record_enabled and self._record_output:
            Path(self._record_output).expanduser().parent.mkdir(parents=True, exist_ok=True)
            print(
                "[FeatureThenFusionAdapterSensorRigAgent] recording "
                f"sensor={self._record_sensor_id} output={self._record_output}",
                flush=True,
            )
            atexit.register(self._close_recording)

    def _close_recording(self) -> None:
        if self._record_writer is None:
            return
        self._record_writer.release()
        print(
            f"[FeatureThenFusionAdapterSensorRigAgent] saved_frames={self._record_frames} "
            f"output={self._record_output}",
            flush=True,
        )
        self._record_writer = None

    def _write_record_frame(self, input_data: Mapping[str, Any]) -> None:
        if not self._record_enabled or not self._record_output:
            return
        step = int(getattr(self, "step", -1))
        if step >= 0 and step % self._record_every_n != 0:
            return
        if self._record_sensor_id not in input_data:
            return
        frame = input_data[self._record_sensor_id][1][:, :, :3].copy()
        if self._record_scale > 0 and abs(self._record_scale - 1.0) > 1e-6:
            frame = cv2.resize(frame, None, fx=self._record_scale, fy=self._record_scale, interpolation=cv2.INTER_LINEAR)
        height, width = frame.shape[:2]
        if self._record_writer is None:
            self._record_writer = cv2.VideoWriter(
                str(Path(self._record_output).expanduser()),
                cv2.VideoWriter_fourcc(*os.environ.get("TFPP_RECORD_CODEC", "MJPG")),
                self._record_fps,
                (width, height),
            )
            if not self._record_writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {self._record_output}")
        cv2.putText(frame, f"step={step}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        self._record_writer.write(frame)
        self._record_frames += 1

    def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=unused-argument
        self._write_record_frame(input_data)
        return super().run_step(input_data, timestamp, sensors=sensors)

    def destroy(self, results=None):
        try:
            self._close_recording()
        finally:
            return super().destroy(results=results)
