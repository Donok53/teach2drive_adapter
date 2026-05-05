import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


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


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_image(path: Path, size: Tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return np.transpose(image, (2, 0, 1))


def _load_lidar(path: Path, size: int) -> np.ndarray:
    if not path.exists():
        return np.zeros((1, size, size), dtype=np.float32)
    bev = np.load(path).astype(np.float32)
    if bev.ndim == 2:
        bev = bev[None, :, :]
    elif bev.ndim == 3 and bev.shape[-1] <= 8:
        bev = np.transpose(bev, (2, 0, 1))
    if bev.shape[-2:] != (size, size):
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
    ) -> None:
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
                images.append(np.zeros((3, self.image_size[1], self.image_size[0]), dtype=np.float32))
            else:
                images.append(_load_image(episode_dir / token, self.image_size))
        camera_tensor = np.stack(images, axis=0)

        lidar_token = frame.get("lidar_bev_token")
        if lidar_token:
            lidar = _load_lidar(episode_dir / lidar_token, self.lidar_size)
        else:
            lidar = np.zeros((1, self.lidar_size, self.lidar_size), dtype=np.float32)

        target = np.concatenate([self.traj[idx], self.speed[idx], self.stop[idx]], axis=0).astype(np.float32)
        return {
            "scalar": torch.from_numpy(self.scalar[idx]),
            "camera": torch.from_numpy(camera_tensor),
            "lidar": torch.from_numpy(lidar),
            "target": torch.from_numpy(target),
            "stop_state": torch.tensor(self.stop_state[idx], dtype=torch.long),
            "stop_reason": torch.tensor(self.stop_reason[idx], dtype=torch.long),
            "stop_reason_mask": torch.from_numpy(self.stop_reason_mask[idx]),
            "sample_weight": torch.from_numpy(self.sample_weight[idx]),
        }


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

