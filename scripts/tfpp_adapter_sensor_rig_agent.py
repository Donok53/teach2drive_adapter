#!/usr/bin/env python3
"""TransFuser++ leaderboard agent with shifted sensors and a trained adapter."""

from __future__ import annotations

import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import cv2
import carla
import numpy as np
import torch
import torch.nn.functional as F

from sensor_agent import SensorAgent
import transfuser_utils as t_u


def get_entry_point() -> str:
    return "AdapterSensorRigAgent"


_BUILTIN_RIGS: dict[str, dict[str, Any]] = {
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
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


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


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _csv(value: str | None, default: str) -> list[str]:
    raw = default if value is None or not value.strip() else value
    return [item.strip() for item in raw.split(",") if item.strip()]


def _pose_vec(pose: Mapping[str, Any], keys: tuple[str, str, str]) -> list[float]:
    return [float(pose[key]) for key in keys]


def _load_layout() -> tuple[str, Mapping[str, Any]]:
    layout_path = os.environ.get("TFPP_SENSOR_RIG_JSON")
    if layout_path:
        path = Path(layout_path).expanduser()
        return str(path), json.loads(path.read_text(encoding="utf-8"))
    rig = os.environ.get("TFPP_SENSOR_RIG", "front_triplet_shifted")
    if rig not in _BUILTIN_RIGS:
        raise ValueError(f"Unknown TFPP_SENSOR_RIG={rig!r}. Available: {sorted(_BUILTIN_RIGS)}")
    return rig, _BUILTIN_RIGS[rig]


def _ensure_adapter_import_path() -> None:
    root = Path(os.environ.get("ADAPTER_ROOT", "/home/byeongjae/code/teach2drive_adapter")).expanduser()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _camera_sensor_from_pose(agent: SensorAgent, name: str, pose: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "sensor.camera.rgb",
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "z": float(pose["z"]),
        "roll": float(pose["roll"]),
        "pitch": float(pose["pitch"]),
        "yaw": float(pose["yaw"]),
        "width": int(agent.config.camera_width),
        "height": int(agent.config.camera_height),
        "fov": float(agent.config.camera_fov),
        "id": f"rgb_{name}",
    }


def _resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (image.shape[1], image.shape[0]) == size:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _rgb_from_input(input_data: Mapping[str, Any], sensor_id: str) -> np.ndarray:
    # CARLA leaderboard delivers BGRA/BGR-like arrays; match CARLA Garage preprocessing.
    bgr = input_data[sensor_id][1][:, :, :3]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _adapter_lidar_bev(points: np.ndarray, size: int = 128) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.zeros((1, size, size), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[(pts[:, 2] > -100.0) & (pts[:, 2] < 100.0)]
    pts = pts[pts[:, 2] > 0.2]
    if len(pts) == 0:
        return np.zeros((1, size, size), dtype=np.float32)
    xbins = np.linspace(-32.0, 32.0, size + 1)
    ybins = np.linspace(-32.0, 32.0, size + 1)
    hist = np.histogramdd(pts[:, :2], bins=(xbins, ybins))[0]
    hist = np.minimum(hist, 5.0) / 5.0
    return hist.T[np.newaxis].astype(np.float32)


class AdapterSensorRigAgent(SensorAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)
        self._rig_source, self._rig_layout = _load_layout()
        self._adapter_camera_names = _csv(os.environ.get("TFPP_ADAPTER_CAMERAS"), "left,front,right")
        self._apply_sensor_rig()
        self._setup_adapter()
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
            "[AdapterSensorRigAgent] applied rig="
            f"{self._rig_source} camera={camera_name} pos={self.config.camera_pos} rot={self.config.camera_rot_0} "
            f"lidar={lidar_name} pos={self.config.lidar_pos} rot={self.config.lidar_rot}",
            flush=True,
        )

    def _layout_tensor(self, layout_dim: int) -> torch.Tensor:
        _ensure_adapter_import_path()
        from teach2drive_adapter.sensor_layout import CameraSpec, LidarSpec, Pose6D, SensorLayout, flatten_sensor_layout

        cameras = {
            name: CameraSpec(
                Pose6D(
                    x=float(spec.get("x", 0.0)),
                    y=float(spec.get("y", 0.0)),
                    z=float(spec.get("z", 0.0)),
                    roll=float(spec.get("roll", 0.0)),
                    pitch=float(spec.get("pitch", 0.0)),
                    yaw=float(spec.get("yaw", 0.0)),
                ),
                fov=float(spec.get("fov", self.config.camera_fov)),
                width=int(spec.get("width", self.config.camera_width)),
                height=int(spec.get("height", self.config.camera_height)),
                present=True,
            )
            for name, spec in self._rig_layout.get("cameras", {}).items()
        }
        lidars = {
            name: LidarSpec(
                Pose6D(
                    x=float(spec.get("x", 0.0)),
                    y=float(spec.get("y", 0.0)),
                    z=float(spec.get("z", 0.0)),
                    roll=float(spec.get("roll", 0.0)),
                    pitch=float(spec.get("pitch", 0.0)),
                    yaw=float(spec.get("yaw", 0.0)),
                ),
                channels=int(spec.get("channels", 64)),
                range=float(spec.get("range", 85.0)),
                present=True,
            )
            for name, spec in self._rig_layout.get("lidars", {}).items()
        }
        layout = flatten_sensor_layout(SensorLayout(cameras=cameras, lidars=lidars, estimated=False))
        fixed = np.zeros((layout_dim,), dtype=np.float32)
        fixed[: min(layout_dim, layout.shape[0])] = layout[: min(layout_dim, layout.shape[0])]
        return torch.from_numpy(fixed).to(self.device, dtype=torch.float32).unsqueeze(0)

    def _setup_adapter(self) -> None:
        checkpoint_path = os.environ.get("TFPP_ADAPTER_CHECKPOINT", "")
        if not checkpoint_path:
            raise ValueError("TFPP_ADAPTER_CHECKPOINT is required for AdapterSensorRigAgent")
        _ensure_adapter_import_path()
        from teach2drive_adapter.train_transfuserpp_cached_visual_adapter import CachedVisualTransFuserPPAdapterPolicy

        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=self.device)
        ckpt_args = checkpoint.get("args", {})
        self._adapter_target_dim = int(checkpoint.get("target_dim", 17))
        self._adapter_speed_dim = int(ckpt_args.get("speed_dim", 4))
        self._adapter_checkpoint_dim = int(checkpoint.get("checkpoint_dim", 20))
        self._adapter_speed_classes = int(checkpoint.get("speed_classes", len(getattr(self.config, "target_speeds", [])) or 8))
        self._adapter_visual_size = tuple(int(v) for v in ckpt_args.get("image_size", [320, 180]))
        self._adapter_lidar_size = int(ckpt_args.get("lidar_size", 128))
        self._adapter_layout = self._layout_tensor(int(checkpoint.get("layout_dim", 52)))
        self._adapter_use_stop = _truthy(os.environ.get("TFPP_ADAPTER_USE_STOP"), default=False)
        self._adapter_stop_threshold = _env_float("TFPP_ADAPTER_STOP_THRESHOLD", 0.5)
        self._adapter_speed_mode = _env_str("TFPP_ADAPTER_SPEED_MODE", "adapter").lower()
        self._adapter_speed_blend = _env_float("TFPP_ADAPTER_SPEED_BLEND", 0.5)
        self._adapter_control_mode = _env_str("TFPP_ADAPTER_CONTROL_MODE", "pid").lower()
        self._adapter_debug_every = _env_int("TFPP_ADAPTER_DEBUG_EVERY", 40)
        self._last_adapter_debug = {}

        self._adapter = CachedVisualTransFuserPPAdapterPolicy(
            scalar_dim=int(checkpoint.get("scalar_dim", 20)),
            layout_dim=int(checkpoint.get("layout_dim", 52)),
            target_dim=self._adapter_target_dim,
            checkpoint_dim=self._adapter_checkpoint_dim,
            speed_classes=self._adapter_speed_classes,
            cameras=checkpoint.get("cameras", self._adapter_camera_names),
            lidar_channels=int(checkpoint.get("lidar_channels", 1)),
            hidden_dim=int(ckpt_args.get("hidden_dim", 512)),
            layout_hidden_dim=int(ckpt_args.get("layout_hidden_dim", 128)),
            visual_dim=int(ckpt_args.get("visual_dim", 256)),
            visual_token_dim=int(ckpt_args.get("visual_token_dim", 192)),
            visual_layers=int(ckpt_args.get("visual_layers", 2)),
            visual_heads=int(ckpt_args.get("visual_heads", 4)),
            control_dim=int(checkpoint.get("control_dim", 0)),
        ).to(self.device)
        missing, unexpected = self._adapter.load_state_dict(checkpoint["model_state"], strict=False)
        self._adapter.eval()
        print(
            "[AdapterSensorRigAgent] loaded adapter "
            f"checkpoint={Path(checkpoint_path).expanduser()} missing={len(missing)} unexpected={len(unexpected)} "
            f"cameras={checkpoint.get('cameras', self._adapter_camera_names)} "
            f"speed_mode={self._adapter_speed_mode} control_mode={self._adapter_control_mode}",
            flush=True,
        )

    def _setup_recording(self) -> None:
        self._record_enabled = _truthy(os.environ.get("TFPP_AGENT_RECORD_VIDEO"), default=True)
        self._record_sensor_id = os.environ.get("TFPP_RECORD_SENSOR_ID", "rgb_front")
        self._record_fps = _env_float("TFPP_RECORD_FPS", 20.0)
        self._record_output = os.environ.get("TFPP_RECORD_OUTPUT") or os.environ.get("VIDEO_OUTPUT", "")
        self._record_every_n = max(1, _env_int("TFPP_RECORD_EVERY_N", 1))
        self._record_scale = _env_float("TFPP_RECORD_SCALE", 1.0)
        self._record_writer = None
        self._record_frames = 0
        if self._record_enabled and self._record_output:
            Path(self._record_output).expanduser().parent.mkdir(parents=True, exist_ok=True)
            print(f"[AdapterSensorRigAgent] recording sensor={self._record_sensor_id} output={self._record_output}", flush=True)

    def sensors(self):
        sensors = super().sensors()
        existing = {sensor.get("id") for sensor in sensors}
        for name in self._adapter_camera_names:
            sensor_id = f"rgb_{name}"
            if sensor_id in existing:
                continue
            pose = self._rig_layout.get("cameras", {}).get(name)
            if pose is None:
                raise ValueError(f"Adapter camera {name!r} not found in rig {self._rig_source!r}")
            sensors.append(_camera_sensor_from_pose(self, name, pose))
        return sensors

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
                cv2.VideoWriter_fourcc(*os.environ.get("TFPP_RECORD_CODEC", "mp4v")),
                self._record_fps,
                (width, height),
            )
            if not self._record_writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {self._record_output}")
        cv2.putText(frame, f"step={step}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        self._record_writer.write(frame)
        self._record_frames += 1

    def _live_scalar(self, tick_data: Mapping[str, Any], input_data: Mapping[str, Any]) -> torch.Tensor:
        speed = float(tick_data["speed"].detach().cpu().reshape(-1)[0])
        imu = np.asarray(input_data["imu"][1], dtype=np.float32)
        gyro_z = float(imu[5]) if imu.shape[0] > 5 else 0.0
        target = tick_data["target_point"].detach().cpu().numpy().reshape(-1)
        gx = float(target[0]) if target.shape[0] > 0 else 0.0
        gy = float(target[1]) if target.shape[0] > 1 else 0.0
        gyaw = math.atan2(gy, max(gx, 1e-3))
        features = np.asarray(
            [
                speed,
                gyro_z,
                *(imu[:6].tolist() if imu.shape[0] >= 6 else np.pad(imu, (0, max(0, 6 - imu.shape[0])))[:6].tolist()),
                1.0,
                1.0,
                gx,
                gy,
                math.sin(gyaw),
                math.cos(gyaw),
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(features).to(self.device, dtype=torch.float32).unsqueeze(0)

    def _adapter_camera_tensor(self, input_data: Mapping[str, Any]) -> torch.Tensor:
        images = []
        for name in self._adapter_camera_names:
            rgb = _rgb_from_input(input_data, f"rgb_{name}")
            rgb = _resize_rgb(rgb, self._adapter_visual_size)
            images.append(rgb.astype(np.float32).transpose(2, 0, 1) / 255.0)
        return torch.from_numpy(np.stack(images, axis=0)).to(self.device, dtype=torch.float32).unsqueeze(0)

    def _adapt_prediction(self, input_data, tick_data, pred_checkpoint, pred_speed_logits, official_target_speed, lidar_points):
        _ensure_adapter_import_path()
        from teach2drive_adapter.transfuserpp_bridge import base_target_from_checkpoint, speed_expectation

        scalar = self._live_scalar(tick_data, input_data)
        pred_checkpoint_b = pred_checkpoint.unsqueeze(0)
        pred_speed_logits_b = pred_speed_logits.unsqueeze(0)
        base_target = base_target_from_checkpoint(
            pred_checkpoint_b,
            pred_speed_logits_b,
            scalar,
            self.config,
            target_dim=self._adapter_target_dim,
            speed_dim=self._adapter_speed_dim,
        )
        checkpoint_flat = torch.zeros((1, self._adapter_checkpoint_dim), dtype=scalar.dtype, device=self.device)
        raw_checkpoint = pred_checkpoint_b.reshape(1, -1)
        checkpoint_flat[:, : min(self._adapter_checkpoint_dim, raw_checkpoint.shape[1])] = raw_checkpoint[:, : self._adapter_checkpoint_dim]
        speed_logits = torch.zeros((1, self._adapter_speed_classes), dtype=scalar.dtype, device=self.device)
        speed_logits[:, : min(self._adapter_speed_classes, pred_speed_logits_b.shape[1])] = pred_speed_logits_b[:, : self._adapter_speed_classes]
        expected_speed = speed_expectation(pred_speed_logits_b, self.config, 1, self.device)
        camera = self._adapter_camera_tensor(input_data)
        lidar = torch.from_numpy(_adapter_lidar_bev(lidar_points, self._adapter_lidar_size)).to(self.device, dtype=torch.float32).unsqueeze(0)
        expected_channels = int(self._adapter.visual.lidar_encoder.net[0].in_channels)
        if lidar.shape[1] > expected_channels:
            lidar = lidar[:, :expected_channels]
        elif lidar.shape[1] < expected_channels:
            pad = torch.zeros((1, expected_channels - lidar.shape[1], lidar.shape[2], lidar.shape[3]), dtype=lidar.dtype, device=lidar.device)
            lidar = torch.cat([lidar, pad], dim=1)

        out = self._adapter(scalar, self._adapter_layout, base_target, checkpoint_flat, speed_logits, expected_speed, camera, lidar)
        target = out["target"][0]
        traj_dim = int(target.numel()) - self._adapter_speed_dim - 1
        traj = target[:traj_dim].reshape(-1, 3)[:, :2].detach().cpu().numpy()
        speeds = target[traj_dim : traj_dim + self._adapter_speed_dim].detach().cpu().numpy()
        stop_prob = float(torch.sigmoid(target[-1]).detach().cpu())
        adapter_speed = max(float(speeds[min(1, len(speeds) - 1)]) if len(speeds) else official_target_speed, 0.0)
        direct_control = None
        if "control" in out:
            raw_control = out["control"][0].detach().cpu().numpy()
            direct_control = (
                float(np.clip(raw_control[0], -1.0, 1.0)),
                float(np.clip(raw_control[1], 0.0, 1.0)),
                float(np.clip(raw_control[2], 0.0, 1.0)),
            )
        if self._adapter_speed_mode == "official":
            target_speed = official_target_speed
        elif self._adapter_speed_mode == "blend":
            target_speed = self._adapter_speed_blend * adapter_speed + (1.0 - self._adapter_speed_blend) * official_target_speed
        else:
            target_speed = adapter_speed
        if self._adapter_use_stop and stop_prob >= self._adapter_stop_threshold:
            target_speed = 0.0
        self._last_adapter_debug = {
            "official_target_speed": float(official_target_speed),
            "adapter_speed": float(adapter_speed),
            "target_speed": float(target_speed),
            "speed_horizons": [float(v) for v in speeds.tolist()],
            "stop_prob": float(stop_prob),
            "direct_control": direct_control,
        }
        return traj, float(target_speed), stop_prob, direct_control

    @torch.inference_mode()
    def run_step(self, input_data, timestamp, sensors=None):  # pylint: disable=unused-argument
        self._write_record_frame(input_data)
        self.step += 1

        if not self.initialized:
            self._init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            tick_data = self.tick(input_data)
            if self.config.backbone not in ("aim"):
                self.lidar_last = deepcopy(tick_data["lidar"])
            return control

        tick_data = self.tick(input_data)
        lidar_indices = [i * self.config.data_save_freq for i in range(self.config.lidar_seq_len)]

        ego_x, ego_y, ego_theta = self.state_log[-1][0], self.state_log[-1][1], self.state_log[-1][2]
        ego_x_last, ego_y_last, ego_theta_last = self.state_log[-2][0], self.state_log[-2][1], self.state_log[-2][2]

        if self.config.backbone not in ("aim"):
            lidar_last = self.align_lidar(self.lidar_last, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

        if self.stop_sign_controller:
            self.update_stop_box(self.stop_sign_buffer, ego_x_last, ego_y_last, ego_theta_last, ego_x, ego_y, ego_theta)

        lidar_full_for_adapter = None
        if self.config.backbone not in ("aim"):
            lidar_current = deepcopy(tick_data["lidar"])
            lidar_full = np.concatenate((lidar_current, lidar_last), axis=0)
            lidar_full_for_adapter = lidar_full
            self.lidar_buffer.append(lidar_full)
            if len(self.lidar_buffer) < (self.config.lidar_seq_len * self.config.data_save_freq):
                self.lidar_last = deepcopy(tick_data["lidar"])
                tmp_control = carla.VehicleControl(0.0, 0.0, 1.0)
                self.control = tmp_control
                return tmp_control

        if self.config.backbone in ("aim"):
            lidar_bev = torch.zeros(
                (1, 1 + int(self.config.use_ground_plane), self.config.lidar_resolution_height, self.config.lidar_resolution_width),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            lidar_bev = []
            for i in lidar_indices:
                lidar_point_cloud = deepcopy(self.lidar_buffer[-(i + 1)])
                if self.config.realign_lidar and self.config.lidar_seq_len > 1:
                    curr_x = self.state_log[i][0]
                    curr_y = self.state_log[i][1]
                    curr_theta = self.state_log[i][2]
                    lidar_point_cloud = self.align_lidar(lidar_point_cloud, curr_x, curr_y, curr_theta, ego_x, ego_y, ego_theta)
                lidar_histogram = self.data.lidar_to_histogram_features(lidar_point_cloud, use_ground_plane=self.config.use_ground_plane)
                lidar_bev.append(torch.from_numpy(lidar_histogram).unsqueeze(0).to(self.device, dtype=torch.float32))
            lidar_bev = torch.cat(lidar_bev, dim=1)
            self.lidar_last = deepcopy(tick_data["lidar"])

        gt_velocity = tick_data["speed"]
        velocity = gt_velocity.reshape(1, 1)
        speed = gt_velocity.item()

        if self.stop_after_meter > 0:
            self.meters_travelled += speed * self.config.carla_frame_rate

        pred_target_speed_probs = []
        pred_target_speed_logits = []
        pred_checkpoints = []
        bounding_boxes = []
        pred_wp = pred_semantic = pred_bev_semantic = pred_depth = pred_bb_features = pred_target_speed = None
        selected_path = attention_weights = pred_wp_1 = wp_selected = None
        compute_debug_output = self.config.debug and (self.save_path is not None)

        for i in range(self.model_count):
            if self.config.backbone not in ("transFuser", "aim", "bev_encoder"):
                raise ValueError("The chosen vision backbone does not exist.")
            (
                pred_wp,
                pred_target_speed,
                pred_checkpoint,
                pred_semantic,
                pred_bev_semantic,
                pred_depth,
                pred_bb_features,
                attention_weights,
                pred_wp_1,
                selected_path,
            ) = self.nets[i].forward(
                rgb=tick_data["rgb"],
                lidar_bev=lidar_bev,
                target_point=tick_data["target_point"],
                target_point_next=tick_data["target_point_next"] if self.config.two_tp_input else None,
                ego_vel=velocity,
                command=tick_data["command"],
            )
            if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ("aim") or self.stop_sign_controller):
                pred_bounding_box = self.nets[i].convert_features_to_bb_metric(pred_bb_features)
            else:
                pred_bounding_box = None
            if self.config.use_wp_gru:
                if self.config.multi_wp_output:
                    wp_selected = 1 if F.sigmoid(selected_path)[0].item() > 0.5 else 0
                    pred_w = pred_wp_1 if wp_selected else pred_wp
                else:
                    pred_w = pred_wp
                self.pred_wp = pred_w if i == 0 else self.pred_wp + pred_w
            if self.config.use_controller_input_prediction:
                pred_target_speed_logits.append(pred_target_speed[0])
                pred_target_speed_probs.append(F.softmax(pred_target_speed[0], dim=0))
                pred_checkpoints.append(pred_checkpoint[0])
            bounding_boxes.append(pred_bounding_box)

        if self.config.detect_boxes and (compute_debug_output or self.config.backbone in ("aim") or self.stop_sign_controller):
            bbs_vehicle_coordinate_system = t_u.non_maximum_suppression(bounding_boxes, self.config.iou_treshold_nms)
            self.bb_buffer.append(bbs_vehicle_coordinate_system)
        else:
            bbs_vehicle_coordinate_system = None

        if self.stop_sign_controller:
            stop_for_stop_sign = self.stop_sign_controller_step(gt_velocity.item())
        else:
            stop_for_stop_sign = False

        if self.config.tp_attention:
            self.tp_attention_buffer.append(attention_weights[2])

        if self.config.use_wp_gru:
            self.pred_wp = self.pred_wp / max(self.model_count, 1)

        if self.config.use_controller_input_prediction:
            pred_target_speed_ensemble = torch.stack(pred_target_speed_probs, dim=0).mean(dim=0)
            pred_speed_logits_ensemble = torch.stack(pred_target_speed_logits, dim=0).mean(dim=0)
            pred_checkpoints_ensemble = torch.stack(pred_checkpoints, dim=0).mean(dim=0)
            if self.uncertainty_weight:
                uncertainty = pred_target_speed_ensemble.detach().cpu().numpy()
                if uncertainty[0] > self.config.brake_uncertainty_threshold:
                    pred_target_speed_scalar = self.inference_target_speeds[0]
                else:
                    pred_target_speed_scalar = sum(uncertainty * self.inference_target_speeds)
            else:
                pred_target_speed_scalar = self.inference_target_speeds[torch.argmax(pred_target_speed_ensemble)]

        adapter_stop_prob = -1.0
        control_source = "pid"
        if self.config.inference_direct_controller and self.config.use_controller_input_prediction:
            adapted_checkpoints, adapted_target_speed, adapter_stop_prob, adapter_direct_control = self._adapt_prediction(
                input_data,
                tick_data,
                pred_checkpoints_ensemble,
                pred_speed_logits_ensemble,
                float(pred_target_speed_scalar),
                lidar_full_for_adapter,
            )
            if self._adapter_control_mode in {"direct", "direct_longitudinal", "hybrid_longitudinal"} and adapter_direct_control is not None:
                steer, throttle, brake = adapter_direct_control
                if self._adapter_control_mode in {"direct_longitudinal", "hybrid_longitudinal"}:
                    pid_steer, _, _ = self.nets[0].control_pid_direct(adapted_checkpoints, adapted_target_speed, gt_velocity)
                    steer = pid_steer
                control_source = "direct"
            else:
                steer, throttle, brake = self.nets[0].control_pid_direct(adapted_checkpoints, adapted_target_speed, gt_velocity)
        elif self.config.use_wp_gru and not self.config.inference_direct_controller:
            steer, throttle, brake = self.nets[0].control_pid(
                self.pred_wp,
                gt_velocity,
                tuned_aim_distance=bool(self.tuned_aim_distance),
            )
        else:
            raise ValueError("An output representation was chosen that was not trained.")

        override_reason = "none"
        if self.step % max(1, int(self._adapter_debug_every)) == 0:
            dbg = getattr(self, "_last_adapter_debug", {})
            print(
                f"[AdapterSensorRigAgent] step={self.step} speed={speed:.2f} "
                f"mode={self._adapter_control_mode}/{control_source} "
                f"official_speed={float(dbg.get('official_target_speed', 0.0)):.2f} "
                f"adapter_speed={float(dbg.get('adapter_speed', 0.0)):.2f} "
                f"adapter_target_speed={float(dbg.get('target_speed', locals().get('adapted_target_speed', 0.0))):.2f} "
                f"speeds={dbg.get('speed_horizons', [])} "
                f"stop={adapter_stop_prob:.2f} throttle={float(locals().get('throttle', 0.0)):.2f} "
                f"brake={float(locals().get('brake', 0.0)):.2f}",
                flush=True,
            )

        if gt_velocity < 0.1:
            self.stuck_detector += 1
        else:
            self.stuck_detector = 0
        if self.stuck_detector > self.config.stuck_threshold:
            self.force_move = self.config.creep_duration
        if self.force_move > 0:
            emergency_stop = False
            if self.config.backbone not in ("aim"):
                safety_box = deepcopy(self.lidar_buffer[-1])
                safety_box = safety_box[safety_box[..., 2] > self.config.safety_box_z_min]
                safety_box = safety_box[safety_box[..., 2] < self.config.safety_box_z_max]
                safety_box = safety_box[safety_box[..., 1] > self.config.safety_box_y_min]
                safety_box = safety_box[safety_box[..., 1] < self.config.safety_box_y_max]
                safety_box = safety_box[safety_box[..., 0] > self.config.safety_box_x_min]
                safety_box = safety_box[safety_box[..., 0] < self.config.safety_box_x_max]
                emergency_stop = len(safety_box) > 0
            if not emergency_stop:
                print("Detected agent being stuck. Step: ", self.step)
                throttle = max(self.config.creep_throttle, throttle)
                brake = False
                self.force_move -= 1
                override_reason = "stuck_creep"
            else:
                print("Creeping stopped by safety box. Step: ", self.step)
                throttle = 0.0
                brake = True
                self.force_move = self.config.creep_duration
                override_reason = "safety_box"

        if self.stop_sign_controller and stop_for_stop_sign:
            throttle = 0.0
            brake = True
            override_reason = "stop_sign_controller"
        if self.stop_after_meter > 0 and self.meters_travelled > self.stop_after_meter:
            throttle = 0.0
            brake = True
            override_reason = "stop_after_meter"

        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
        if self.step < self.config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
            override_reason = "initial_delay"
        else:
            self.control = control
        if self.step % max(1, int(self._adapter_debug_every)) == 0:
            print(
                f"[AdapterSensorRigAgent] final_control step={self.step} "
                f"steer={self.control.steer:.3f} throttle={self.control.throttle:.3f} "
                f"brake={self.control.brake:.3f} override={override_reason}",
                flush=True,
            )
        return self.control

    def destroy(self, results=None):
        try:
            if self._record_writer is not None:
                self._record_writer.release()
                print(f"[AdapterSensorRigAgent] saved_frames={self._record_frames} output={self._record_output}", flush=True)
        finally:
            self._record_writer = None
            return super().destroy(results=results)
