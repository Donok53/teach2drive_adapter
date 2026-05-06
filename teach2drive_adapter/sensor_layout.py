import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Pose6D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def as_radians(self) -> "Pose6D":
        return Pose6D(
            x=self.x,
            y=self.y,
            z=self.z,
            roll=math.radians(self.roll),
            pitch=math.radians(self.pitch),
            yaw=math.radians(self.yaw),
        )


@dataclass(frozen=True)
class CameraSpec:
    pose: Pose6D
    fov: float = 120.0
    width: int = 640
    height: int = 360
    present: bool = True


@dataclass(frozen=True)
class LidarSpec:
    pose: Pose6D
    channels: int = 64
    range: float = 85.0
    present: bool = True


@dataclass(frozen=True)
class SensorLayout:
    cameras: Mapping[str, CameraSpec]
    lidars: Mapping[str, LidarSpec]
    estimated: bool = False


CANONICAL_CAMERA_ORDER = ("left", "front", "right")
CANONICAL_LIDAR_ORDER = ("top",)


def canonical_transfuserpp_layout() -> SensorLayout:
    cameras = {
        "left": CameraSpec(Pose6D(x=1.3, y=-0.4, z=2.3, yaw=-60.0)),
        "front": CameraSpec(Pose6D(x=1.3, y=0.0, z=2.3, yaw=0.0)),
        "right": CameraSpec(Pose6D(x=1.3, y=0.4, z=2.3, yaw=60.0)),
    }
    lidars = {"top": LidarSpec(Pose6D(x=1.3, y=0.0, z=2.5, yaw=0.0))}
    return SensorLayout(cameras=cameras, lidars=lidars, estimated=True)


def teach2drive_tokens_layout() -> SensorLayout:
    cameras = {
        "left": CameraSpec(Pose6D(x=1.0, y=-0.55, z=1.55, yaw=-45.0), fov=90.0, width=640, height=360),
        "front": CameraSpec(Pose6D(x=1.5, y=0.0, z=1.6, yaw=0.0), fov=90.0, width=640, height=360),
        "right": CameraSpec(Pose6D(x=1.0, y=0.55, z=1.55, yaw=45.0), fov=90.0, width=640, height=360),
    }
    lidars = {"top": LidarSpec(Pose6D(x=0.0, y=0.0, z=1.8), channels=32, range=60.0)}
    return SensorLayout(cameras=cameras, lidars=lidars, estimated=False)


def _pose_from_mapping(raw: Mapping) -> Pose6D:
    return Pose6D(
        x=float(raw.get("x", 0.0)),
        y=float(raw.get("y", 0.0)),
        z=float(raw.get("z", 0.0)),
        roll=float(raw.get("roll", 0.0)),
        pitch=float(raw.get("pitch", 0.0)),
        yaw=float(raw.get("yaw", 0.0)),
    )


def _camera_from_mapping(raw: Mapping) -> CameraSpec:
    pose = _pose_from_mapping(raw)
    return CameraSpec(
        pose=pose,
        fov=float(raw.get("fov", 120.0)),
        width=int(raw.get("width", 640)),
        height=int(raw.get("height", 360)),
        present=bool(raw.get("present", True)),
    )


def _lidar_from_mapping(raw: Mapping) -> LidarSpec:
    pose = _pose_from_mapping(raw)
    return LidarSpec(
        pose=pose,
        channels=int(raw.get("channels", 64)),
        range=float(raw.get("range", 85.0)),
        present=bool(raw.get("present", True)),
    )


def load_sensor_layout(episode_dir: Path, file_name: str = "sensor_layout.json") -> SensorLayout:
    path = Path(episode_dir) / file_name
    if not path.exists():
        return canonical_transfuserpp_layout()
    raw = json.loads(path.read_text(encoding="utf-8"))
    cameras = {name: _camera_from_mapping(spec) for name, spec in raw.get("cameras", {}).items()}
    lidars = {name: _lidar_from_mapping(spec) for name, spec in raw.get("lidars", {}).items()}
    return SensorLayout(cameras=cameras, lidars=lidars, estimated=bool(raw.get("estimated", False)))


def save_sensor_layout(layout: SensorLayout, path: Path) -> None:
    raw = {
        "estimated": layout.estimated,
        "cameras": {
            name: {
                **asdict(spec.pose),
                "fov": spec.fov,
                "width": spec.width,
                "height": spec.height,
                "present": spec.present,
            }
            for name, spec in layout.cameras.items()
        },
        "lidars": {
            name: {
                **asdict(spec.pose),
                "channels": spec.channels,
                "range": spec.range,
                "present": spec.present,
            }
            for name, spec in layout.lidars.items()
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(raw, indent=2), encoding="utf-8")


def _pose_features(pose: Pose6D) -> Sequence[float]:
    pose_rad = pose.as_radians()
    return (
        pose_rad.x,
        pose_rad.y,
        pose_rad.z,
        math.sin(pose_rad.roll),
        math.cos(pose_rad.roll),
        math.sin(pose_rad.pitch),
        math.cos(pose_rad.pitch),
        math.sin(pose_rad.yaw),
        math.cos(pose_rad.yaw),
    )


def flatten_sensor_layout(
    layout: SensorLayout,
    camera_order: Sequence[str] = CANONICAL_CAMERA_ORDER,
    lidar_order: Sequence[str] = CANONICAL_LIDAR_ORDER,
) -> np.ndarray:
    features = []
    for name in camera_order:
        spec = layout.cameras.get(name)
        if spec is None:
            features.extend([0.0] * 13)
            continue
        features.extend(_pose_features(spec.pose))
        features.extend(
            [
                float(spec.fov) / 180.0,
                float(spec.width) / 2048.0,
                float(spec.height) / 2048.0,
                1.0 if spec.present else 0.0,
            ]
        )
    for name in lidar_order:
        spec = layout.lidars.get(name)
        if spec is None:
            features.extend([0.0] * 12)
            continue
        features.extend(_pose_features(spec.pose))
        features.extend([float(spec.channels) / 128.0, float(spec.range) / 150.0, 1.0 if spec.present else 0.0])
    features.append(1.0 if layout.estimated else 0.0)
    return np.asarray(features, dtype=np.float32)


def perturb_layout(
    layout: SensorLayout,
    camera_yaw_deg: float = 0.0,
    camera_pitch_deg: float = 0.0,
    camera_z_m: float = 0.0,
    lidar_yaw_deg: float = 0.0,
    lidar_z_m: float = 0.0,
) -> SensorLayout:
    cameras: MutableMapping[str, CameraSpec] = {}
    for name, spec in layout.cameras.items():
        pose = Pose6D(
            x=spec.pose.x,
            y=spec.pose.y,
            z=spec.pose.z + camera_z_m,
            roll=spec.pose.roll,
            pitch=spec.pose.pitch + camera_pitch_deg,
            yaw=spec.pose.yaw + camera_yaw_deg,
        )
        cameras[name] = CameraSpec(pose=pose, fov=spec.fov, width=spec.width, height=spec.height, present=spec.present)
    lidars: MutableMapping[str, LidarSpec] = {}
    for name, spec in layout.lidars.items():
        pose = Pose6D(
            x=spec.pose.x,
            y=spec.pose.y,
            z=spec.pose.z + lidar_z_m,
            roll=spec.pose.roll,
            pitch=spec.pose.pitch,
            yaw=spec.pose.yaw + lidar_yaw_deg,
        )
        lidars[name] = LidarSpec(pose=pose, channels=spec.channels, range=spec.range, present=spec.present)
    return SensorLayout(cameras=cameras, lidars=lidars, estimated=layout.estimated)
