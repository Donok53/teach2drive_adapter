import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .sensor_layout import flatten_sensor_layout, load_sensor_layout


STOP_STATE_NAMES = ["drive", "approach_stop", "stopped_waiting", "release_go"]
STOP_REASON_NAMES = [
    "none",
    "unknown_stop",
    "startup",
    "route_end",
    "traffic_light",
    "stop_sign",
    "front_vehicle",
    "junction_yield",
]

# Controller-input target-speed support used by the released TF++ checkpoint.
TFPP_TARGET_SPEEDS = (0.0, 4.0, 8.0, 10.0, 13.88888889, 16.0, 17.77777777, 20.0)


def _tfpp_two_hot_target_speed(
    target_speed: float,
    brake: bool,
    target_speeds: Sequence[float] = TFPP_TARGET_SPEEDS,
) -> np.ndarray:
    """Reproduce CARLA Garage's TF++ target-speed encoding."""
    values = np.asarray(target_speeds, dtype=np.float32)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("target_speeds must contain at least two ordered values")
    target_speed = float(target_speed)
    if target_speed < 0.0:
        raise ValueError("TF++ target speed must be non-negative")
    label = np.zeros((len(values),), dtype=np.float32)
    if bool(brake):
        label[0] = 1.0
        return label
    larger = np.flatnonzero(values > target_speed)
    if len(larger) == 0:
        label[-1] = 1.0
        return label
    upper = int(larger[0])
    lower = max(upper - 1, 0)
    if upper == lower:
        label[upper] = 1.0
        return label
    width = max(float(values[upper] - values[lower]), 1e-6)
    label[lower] = (float(values[upper]) - target_speed) / width
    label[upper] = (target_speed - float(values[lower])) / width
    return label


def _tfpp_command_one_hot(command: object) -> np.ndarray:
    """Match transfuser_utils.command_to_one_hot without importing CARLA Garage."""
    result = np.zeros((6,), dtype=np.float32)
    try:
        index = int(command) - 1
    except (TypeError, ValueError):
        index = 3  # lane-follow command 4
    if index not in range(6):
        index = 3
    result[index] = 1.0
    return result


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_image(path: Path, size: Tuple[int, int], allow_resize: bool = True) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size and (image.shape[1], image.shape[0]) != tuple(size):
        if not allow_resize:
            raise ValueError(
                f"Exact sensor preprocessing requires image {tuple(size)}, "
                f"but {path} is {(image.shape[1], image.shape[0])}"
            )
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))


# v12 depth GT storage grid (must match scripts/gen_depth_gt.py GH, GW)
DEPTH_GT_HW = (96, 256)


def _load_lidar(path: Path, size: int, allow_resize: bool = True) -> np.ndarray:
    if not path.exists():
        return np.zeros((1, size, size), dtype=np.float32)
    bev = np.load(path).astype(np.float32)
    if bev.ndim == 2:
        bev = bev[None, :, :]
    elif bev.ndim == 3 and bev.shape[-1] <= 8:
        bev = np.transpose(bev, (2, 0, 1))
    if bev.shape[-2:] != (size, size):
        if not allow_resize:
            raise ValueError(
                f"Exact sensor preprocessing requires LiDAR BEV {(size, size)}, "
                f"but {path} is {tuple(bev.shape[-2:])}. Re-run the exact TF++ converter."
            )
        channels = []
        for channel in bev:
            channels.append(cv2.resize(channel, (size, size), interpolation=cv2.INTER_AREA))
        bev = np.stack(channels, axis=0)
    return bev.astype(np.float32)


def _resolve_episode_dirs(raw_dirs: Sequence, override_root: Optional[str]) -> List[Path]:
    dirs = [Path(str(item)) for item in raw_dirs]
    if not override_root:
        return dirs
    root = Path(override_root).expanduser().resolve()
    return [root / path.name for path in dirs]


def _resample_tfpp_route(route: np.ndarray, count: int) -> np.ndarray:
    """Match TF++ smooth_path spacing: 2.5 m first point, then 1 m."""
    count = int(count)
    route = np.asarray(route, dtype=np.float32).reshape(-1, 2)
    points = np.concatenate([np.zeros((1, 2), dtype=np.float32), route], axis=0)
    # Remove duplicate consecutive points so interpolation/extrapolation remains stable.
    keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6])
    points = points[keep]
    if len(points) < 2:
        return np.zeros((count, 2), dtype=np.float32)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segment_lengths)]).astype(np.float32)
    distances = 2.5 + np.arange(count, dtype=np.float32)
    result = np.empty((count, 2), dtype=np.float32)
    for index, distance in enumerate(distances):
        if distance <= arc[-1]:
            segment_index = min(int(np.searchsorted(arc, distance, side="right") - 1), len(points) - 2)
            width = max(float(arc[segment_index + 1] - arc[segment_index]), 1e-6)
            alpha = (float(distance) - float(arc[segment_index])) / width
            result[index] = points[segment_index] * (1.0 - alpha) + points[segment_index + 1] * alpha
        else:
            direction = points[-1] - points[-2]
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            result[index] = points[-1] + direction * (float(distance) - float(arc[-1]))
    return result


def _future_ego_spatial_checkpoints(
    frames: Sequence[Dict],
    frame_index: int,
    count: int,
    first_distance_m: float = 2.5,
    spacing_m: float = 1.0,
    max_horizon_s: float = 6.0,
) -> Tuple[np.ndarray, float]:
    """Build TF++-spaced checkpoints from the expert's future driven path.

    Positions in ``frames.jsonl`` are CARLA world coordinates. Future positions
    are transformed into the current ego frame and interpolated by travelled arc
    length, which removes policy timing differences from the supervision. No
    extrapolation is performed: samples that do not cover the last requested
    checkpoint within ``max_horizon_s`` are marked invalid.
    """
    count = int(count)
    zero = np.zeros((count, 2), dtype=np.float32)
    if count <= 0 or frame_index < 0 or frame_index >= len(frames):
        return zero, 0.0

    current = frames[int(frame_index)]
    odom = current.get("odom", {})
    required = ("x", "y", "yaw")
    if any(key not in odom for key in required):
        return zero, 0.0

    current_x = float(odom["x"])
    current_y = float(odom["y"])
    current_yaw = float(odom["yaw"])
    current_time = float(current.get("time", 0.0))
    cos_yaw = float(np.cos(current_yaw))
    sin_yaw = float(np.sin(current_yaw))
    local_points = []
    for future in frames[int(frame_index) :]:
        future_time = float(future.get("time", current_time))
        if future_time - current_time > float(max_horizon_s) + 1e-6:
            break
        future_odom = future.get("odom", {})
        if "x" not in future_odom or "y" not in future_odom:
            continue
        dx = float(future_odom["x"]) - current_x
        dy = float(future_odom["y"]) - current_y
        # CARLA yaw=0 points along +world-x. This is the inverse planar ego pose.
        forward = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy
        local_points.append((forward, lateral))

    if len(local_points) < 2:
        return zero, 0.0
    points = np.asarray(local_points, dtype=np.float32)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], segment_lengths > 1e-4])
    points = points[keep]
    if len(points) < 2:
        return zero, 0.0
    arc = np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    distances = float(first_distance_m) + np.arange(count, dtype=np.float32) * float(spacing_m)
    if float(arc[-1]) + 1e-4 < float(distances[-1]):
        return zero, 0.0

    result = np.empty((count, 2), dtype=np.float32)
    for index, distance in enumerate(distances):
        segment_index = min(int(np.searchsorted(arc, distance, side="right") - 1), len(points) - 2)
        width = max(float(arc[segment_index + 1] - arc[segment_index]), 1e-6)
        alpha = (float(distance) - float(arc[segment_index])) / width
        result[index] = points[segment_index] * (1.0 - alpha) + points[segment_index + 1] * alpha
    return result, 1.0


class Teach2DriveIndexDataset(Dataset):
    """Dataset for Teach2Drive token index files.

    The index stores scalar features and supervision arrays, while image/LiDAR
    tensors are loaded lazily from each episode directory.
    """

    def __init__(
        self,
        index_path: str,
        indices: Optional[np.ndarray] = None,
        cameras: Optional[Sequence[str]] = None,
        image_size: Tuple[int, int] = (320, 180),
        lidar_size: int = 128,
        episode_root_override: Optional[str] = None,
        teacher_view_root: Optional[str] = None,
        teacher_view_dirname: str = "rgb_front_teacher_xm15",
        teacher_view_camera: str = "front",
        strict_sensor_geometry: bool = False,
        measurement_root: Optional[str] = None,
        route_target_len: int = 10,
        route_target_source: str = "measurement_route",
        future_ego_max_horizon_s: float = 6.0,
        target_speed_values: Sequence[float] = TFPP_TARGET_SPEEDS,
    ) -> None:
        # v4 geometric-teacher distillation: if teacher_view_root is set, __getitem__
        # additionally loads the reprojected x=-1.5 view for `teacher_view_camera`
        # from  <teacher_view_root>/<source_route>/<teacher_view_dirname>/<step>.jpg
        # and returns it as "camera_teacher" [1,3,H,W]. Disabled (None) by default.
        self.teacher_view_root = Path(teacher_view_root).expanduser() if teacher_view_root else None
        self.teacher_view_dirname = str(teacher_view_dirname)
        self.teacher_view_camera = str(teacher_view_camera)
        self.strict_sensor_geometry = bool(strict_sensor_geometry)
        self.measurement_root = Path(measurement_root).expanduser() if measurement_root else None
        self.route_target_len = int(route_target_len)
        self.route_target_source = str(route_target_source)
        self.future_ego_max_horizon_s = float(future_ego_max_horizon_s)
        self.target_speed_values = tuple(float(value) for value in target_speed_values)
        if self.route_target_len <= 0:
            raise ValueError("route_target_len must be positive")
        if self.route_target_source not in {"measurement_route", "future_ego_path"}:
            raise ValueError(
                "route_target_source must be 'measurement_route' or 'future_ego_path', "
                f"got {self.route_target_source!r}"
            )
        self.index_path = Path(index_path).expanduser()
        arrays = np.load(self.index_path, allow_pickle=True)
        self.scalar = arrays["scalar_features"].astype(np.float32)
        self.traj = arrays["traj_targets"].astype(np.float32)
        self.speed = arrays["speed_targets"].astype(np.float32)
        self.stop = arrays["stop_targets"].astype(np.float32).reshape(-1, 1)
        sample_count = len(self.scalar)

        self.stop_state = arrays["stop_state_targets"].astype(np.int64) if "stop_state_targets" in arrays.files else np.zeros(sample_count, dtype=np.int64)
        self.stop_reason = arrays["stop_reason_targets"].astype(np.int64) if "stop_reason_targets" in arrays.files else np.zeros(sample_count, dtype=np.int64)
        self.stop_reason_mask = arrays["stop_reason_masks"].astype(np.float32).reshape(-1, 1) if "stop_reason_masks" in arrays.files else np.zeros((sample_count, 1), dtype=np.float32)
        self.control = arrays["control_targets"].astype(np.float32) if "control_targets" in arrays.files else np.zeros((sample_count, 3), dtype=np.float32)
        self.control_mask = arrays["control_masks"].astype(np.float32).reshape(-1, 1) if "control_masks" in arrays.files else np.zeros((sample_count, 1), dtype=np.float32)
        self.sample_weight = arrays["sample_weights"].astype(np.float32).reshape(-1, 1) if "sample_weights" in arrays.files else np.ones((sample_count, 1), dtype=np.float32)

        self.sample_episode = arrays["sample_episode_indices"].astype(np.int64)
        self.sample_frame = arrays["sample_frame_indices"].astype(np.int64)
        self.episode_dirs = _resolve_episode_dirs(arrays["episode_dirs"], episode_root_override)
        self.index_cameras = [str(item) for item in arrays["cameras"]]
        self.cameras = list(cameras) if cameras else self.index_cameras
        self.image_size = tuple(int(v) for v in image_size)
        self.lidar_size = int(lidar_size)
        self.indices = np.arange(sample_count, dtype=np.int64) if indices is None else indices.astype(np.int64)
        self.frames = [_read_jsonl(path / "frames.jsonl") for path in self.episode_dirs]
        self.layouts = [flatten_sensor_layout(load_sensor_layout(path)) for path in self.episode_dirs]
        self.spatial_route_targets = None
        self.spatial_route_target_masks = None
        if self.route_target_source == "future_ego_path":
            self.spatial_route_targets = np.zeros(
                (sample_count, self.route_target_len, 2), dtype=np.float32
            )
            self.spatial_route_target_masks = np.zeros((sample_count, 1), dtype=np.float32)
            for sample_idx in range(sample_count):
                episode_idx = int(self.sample_episode[sample_idx])
                frame_idx = int(self.sample_frame[sample_idx])
                target, valid = _future_ego_spatial_checkpoints(
                    self.frames[episode_idx],
                    frame_idx,
                    self.route_target_len,
                    max_horizon_s=self.future_ego_max_horizon_s,
                )
                self.spatial_route_targets[sample_idx] = target
                self.spatial_route_target_masks[sample_idx, 0] = valid

    @property
    def layout_dim(self) -> int:
        return int(self.layouts[0].shape[0]) if self.layouts else 0

    @property
    def scalar_dim(self) -> int:
        return int(self.scalar.shape[1])

    @property
    def traj_dim(self) -> int:
        return int(self.traj.shape[1])

    @property
    def speed_dim(self) -> int:
        return int(self.speed.shape[1])

    @property
    def control_dim(self) -> int:
        return int(self.control.shape[1])

    @property
    def target_dim(self) -> int:
        return self.traj_dim + self.speed_dim

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        episode_idx = int(self.sample_episode[idx])
        frame_idx = int(self.sample_frame[idx])
        episode_dir = self.episode_dirs[episode_idx]
        frame = self.frames[episode_idx][frame_idx]

        images = []
        camera_tokens = frame.get("camera_tokens", {})
        for camera in self.cameras:
            token = camera_tokens.get(camera)
            if token is None:
                if self.strict_sensor_geometry:
                    raise ValueError(
                        f"Exact sensor preprocessing requires camera {camera!r}, "
                        f"but episode {episode_dir.name} frame {frame_idx} has no token"
                    )
                images.append(np.zeros((3, self.image_size[1], self.image_size[0]), dtype=np.float32))
            else:
                images.append(
                    _load_image(
                        episode_dir / token,
                        self.image_size,
                        allow_resize=not self.strict_sensor_geometry,
                    )
                )
        camera_tensor = np.stack(images, axis=0)

        lidar_token = frame.get("lidar_bev_token")
        if lidar_token:
            lidar = _load_lidar(
                episode_dir / lidar_token,
                self.lidar_size,
                allow_resize=not self.strict_sensor_geometry,
            )
        else:
            if self.strict_sensor_geometry:
                raise ValueError(
                    f"Exact sensor preprocessing requires LiDAR, "
                    f"but episode {episode_dir.name} frame {frame_idx} has no token"
                )
            lidar = np.zeros((1, self.lidar_size, self.lidar_size), dtype=np.float32)

        target = np.concatenate([self.traj[idx], self.speed[idx], self.stop[idx]], axis=0).astype(np.float32)

        # v12: lidar-projected sparse depth GT sidecar (0 == invalid / no lidar hit).
        # Generated by scripts/gen_depth_gt.py keyed by the converter's frame step.
        # Missing -> all-zero (masked out in the loss), so this is backward compatible.
        depth_gt = np.zeros(DEPTH_GT_HW, dtype=np.float32)
        step = frame.get("step")
        if step is not None:
            depth_path = episode_dir / "depth_gt" / f"{int(step):06d}.npy"
            if depth_path.exists():
                loaded = np.load(depth_path).astype(np.float32)
                if loaded.shape == DEPTH_GT_HW:
                    depth_gt = loaded

        result_extra = {}
        if self.teacher_view_root is not None:
            src_route = frame.get("source_route")
            step = frame.get("step")
            teacher_img = None
            if src_route is not None and step is not None:
                tpath = (self.teacher_view_root / str(src_route) / self.teacher_view_dirname / f"{int(step):04d}.jpg")
                if tpath.exists():
                    teacher_img = _load_image(
                        tpath,
                        self.image_size,
                        allow_resize=not self.strict_sensor_geometry,
                    )
            if teacher_img is None:
                # fallback: use the real front view so the distill loss is a no-op (self-ref)
                fi = self.cameras.index(self.teacher_view_camera) if self.teacher_view_camera in self.cameras else 0
                teacher_img = images[fi]
            result_extra["camera_teacher"] = torch.from_numpy(teacher_img[None])  # [1,3,H,W]

        # Same-frame TF++ spatial checkpoint labels.  These are intentionally
        # loaded from the raw measurement instead of reusing traj_targets: the
        # latter are temporal ego poses (0.5/1.0/1.5/2.0 s), while TF++ predicts
        # spatial route checkpoints with a different sampling contract.
        route_target = np.zeros((self.route_target_len, 2), dtype=np.float32)
        route_target_mask = np.zeros((1,), dtype=np.float32)
        tfpp_target_speed = np.zeros((1,), dtype=np.float32)
        tfpp_target_speed_twohot = np.zeros((len(self.target_speed_values),), dtype=np.float32)
        tfpp_target_speed_mask = np.zeros((1,), dtype=np.float32)
        tfpp_command = _tfpp_command_one_hot(frame.get("command"))
        hazard = frame.get("hazard", {})
        tfpp_hazard = np.asarray(
            [
                bool(hazard.get("light", False)),
                bool(hazard.get("stop_sign", False)),
                bool(hazard.get("vehicle", False)),
                bool(hazard.get("walker", False)),
            ],
            dtype=np.float32,
        )
        if self.route_target_source == "future_ego_path":
            route_target = self.spatial_route_targets[idx].copy()
            route_target_mask = self.spatial_route_target_masks[idx].copy()
        if self.measurement_root is not None:
            src_route = frame.get("source_route")
            step = frame.get("step")
            if src_route is not None and step is not None:
                measurement_path = (
                    self.measurement_root / str(src_route) / "measurements" / f"{int(step):04d}.json.gz"
                )
                if measurement_path.exists():
                    with gzip.open(measurement_path, "rt", encoding="utf-8") as handle:
                        measurement = json.load(handle)
                    if self.route_target_source == "measurement_route":
                        route = np.asarray(measurement.get("route", []), dtype=np.float32).reshape(-1, 2)
                        if len(route) > 0:
                            route_target = _resample_tfpp_route(route, self.route_target_len)
                            route_target_mask[0] = 1.0
                    if measurement.get("target_speed") is not None:
                        target_speed_value = max(float(measurement["target_speed"]), 0.0)
                        brake = bool(measurement.get("brake", False))
                        tfpp_target_speed[0] = target_speed_value
                        tfpp_target_speed_twohot = _tfpp_two_hot_target_speed(
                            target_speed_value,
                            brake,
                            self.target_speed_values,
                        )
                        tfpp_target_speed_mask[0] = 1.0
                    tfpp_command = _tfpp_command_one_hot(measurement.get("command", frame.get("command")))
                    tfpp_hazard = np.asarray(
                        [
                            bool(measurement.get("light_hazard", False)),
                            bool(measurement.get("stop_sign_hazard", False)),
                            bool(measurement.get("vehicle_hazard", False)),
                            bool(measurement.get("walker_hazard", False)),
                        ],
                        dtype=np.float32,
                    )

        sample = {
            "index": torch.tensor(idx, dtype=torch.long),
            "episode_idx": torch.tensor(episode_idx, dtype=torch.long),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
            "scalar": torch.from_numpy(self.scalar[idx]),
            "camera": torch.from_numpy(camera_tensor),
            "lidar": torch.from_numpy(lidar),
            "target": torch.from_numpy(target),
            "depth_gt": torch.from_numpy(depth_gt),
            "stop_state": torch.tensor(self.stop_state[idx], dtype=torch.long),
            "stop_reason": torch.tensor(self.stop_reason[idx], dtype=torch.long),
            "stop_reason_mask": torch.from_numpy(self.stop_reason_mask[idx]),
            "control_target": torch.from_numpy(self.control[idx]),
            "control_mask": torch.from_numpy(self.control_mask[idx]),
            "sample_weight": torch.from_numpy(self.sample_weight[idx]),
            "layout": torch.from_numpy(self.layouts[episode_idx]),
            "route_target": torch.from_numpy(route_target),
            "route_target_mask": torch.from_numpy(route_target_mask),
            "tfpp_target_speed": torch.from_numpy(tfpp_target_speed),
            "tfpp_target_speed_twohot": torch.from_numpy(tfpp_target_speed_twohot),
            "tfpp_target_speed_mask": torch.from_numpy(tfpp_target_speed_mask),
            "tfpp_command": torch.from_numpy(tfpp_command),
            "tfpp_hazard": torch.from_numpy(tfpp_hazard),
        }
        # only include the key when enabled, so default collate is unaffected otherwise
        if "camera_teacher" in result_extra:
            sample["camera_teacher"] = result_extra["camera_teacher"]
        return sample


def split_by_episode(index_path: str, val_ratio: float = 0.15, seed: int = 41) -> Tuple[np.ndarray, np.ndarray]:
    arrays = np.load(Path(index_path).expanduser(), allow_pickle=True)
    sample_episode = arrays["sample_episode_indices"].astype(np.int64)
    episodes = np.unique(sample_episode)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * val_ratio)))
    val_episodes = set(int(v) for v in episodes[:val_count])
    val_mask = np.asarray([int(ep) in val_episodes for ep in sample_episode], dtype=bool)
    all_indices = np.arange(len(sample_episode), dtype=np.int64)
    return all_indices[~val_mask], all_indices[val_mask]
