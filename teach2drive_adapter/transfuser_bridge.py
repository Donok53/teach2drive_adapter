import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


TRANSFUSER_CAMERA_ORDER = ("left", "front", "right")


@dataclass(frozen=True)
class TransFuserInputSpec:
    image_hw: Tuple[int, int] = (160, 704)
    lidar_hw: Tuple[int, int] = (256, 256)
    lidar_channels: int = 2
    expected_camera_order: Tuple[str, str, str] = TRANSFUSER_CAMERA_ORDER
    expected_camera_fov_deg: float = 120.0


def crop_rgb_like_transfuser(image_hwc: np.ndarray, crop_hw: Tuple[int, int] = (160, 704), crop_shift: int = 0) -> np.ndarray:
    """Crop a horizontally stitched RGB image exactly like TransFuser's agent."""

    crop_h, crop_w = crop_hw
    height, width = image_hwc.shape[:2]
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2 + int(crop_shift)
    if start_y < 0 or start_x < 0 or start_y + crop_h > height or start_x + crop_w > width:
        padded = np.zeros((max(height, crop_h), max(width, crop_w), 3), dtype=image_hwc.dtype)
        padded[:height, :width] = image_hwc
        image_hwc = padded
        height, width = image_hwc.shape[:2]
        start_y = max(0, height // 2 - crop_h // 2)
        start_x = max(0, width // 2 - crop_w // 2 + int(crop_shift))
    cropped = image_hwc[start_y : start_y + crop_h, start_x : start_x + crop_w]
    return np.transpose(cropped.astype(np.float32), (2, 0, 1))


def _load_rgb(path: Path, resize_wh: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize_wh is not None and (image.shape[1], image.shape[0]) != tuple(resize_wh):
        image = cv2.resize(image, tuple(resize_wh), interpolation=cv2.INTER_AREA)
    return image


def stitch_camera_views(
    episode_dir: Path,
    frame: Mapping,
    camera_order: Sequence[str] = TRANSFUSER_CAMERA_ORDER,
    missing_policy: str = "repeat_front",
) -> Tuple[np.ndarray, Dict[str, bool]]:
    """Return an HWC image stitched as left/front/right for TransFuser.

    `missing_policy`:
    - `repeat_front`: use the front image for absent side cameras.
    - `zero`: replace absent cameras with a black image.
    """

    camera_tokens = frame.get("camera_tokens", {})
    loaded: MutableMapping[str, np.ndarray] = {}
    present: Dict[str, bool] = {}
    for camera in camera_order:
        token = camera_tokens.get(camera)
        if token and (episode_dir / token).exists():
            loaded[camera] = _load_rgb(episode_dir / token)
            present[camera] = True
        else:
            present[camera] = False

    front = loaded.get("front")
    if front is None:
        existing = next(iter(loaded.values()), None)
        if existing is None:
            raise FileNotFoundError(f"No camera images found in frame {frame.get('step', '<unknown>')} at {episode_dir}")
        front = existing

    output = []
    for camera in camera_order:
        image = loaded.get(camera)
        if image is None:
            if missing_policy == "repeat_front":
                image = front.copy()
            elif missing_policy == "zero":
                image = np.zeros_like(front)
            else:
                raise ValueError(f"Unknown missing camera policy: {missing_policy}")
        if image.shape[:2] != front.shape[:2]:
            image = cv2.resize(image, (front.shape[1], front.shape[0]), interpolation=cv2.INTER_AREA)
        output.append(image)
    return np.concatenate(output, axis=1), present


def lidar_bev_to_transfuser(bev: np.ndarray, spec: TransFuserInputSpec = TransFuserInputSpec()) -> np.ndarray:
    """Map Teach2Drive BEV tokens to TransFuser's 2-channel 256x256 BEV input.

    This is an approximation when raw point clouds are unavailable. Teach2Drive
    stores occupancy/height/intensity-like channels, while TransFuser normally
    uses above/below-height point histograms generated from raw LiDAR points.
    """

    bev = bev.astype(np.float32)
    if bev.ndim == 2:
        bev = bev[None]
    if bev.ndim == 3 and bev.shape[-1] <= 8 and bev.shape[0] > 8:
        bev = np.transpose(bev, (2, 0, 1))

    if bev.shape[0] >= 2:
        lidar = bev[:2]
    else:
        lidar = np.concatenate([bev[:1], np.zeros_like(bev[:1])], axis=0)

    channels = []
    out_h, out_w = spec.lidar_hw
    for channel in lidar:
        channels.append(cv2.resize(channel, (out_w, out_h), interpolation=cv2.INTER_LINEAR))
    return np.stack(channels, axis=0).astype(np.float32)


def target_point_from_scalar(scalar: np.ndarray) -> np.ndarray:
    """Use Teach2Drive's lookahead token as TransFuser's local command point."""

    if scalar.shape[-1] < 12:
        return np.zeros(2, dtype=np.float32)
    return np.asarray([scalar[10], scalar[11]], dtype=np.float32)


def batch_from_teach2drive_sample(sample: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a dataset sample from Teach2DriveIndexDataset to TransFuser-like tensors."""

    camera = sample["camera"].numpy()
    lidar = sample["lidar"].numpy()
    scalar = sample["scalar"].numpy()

    # Teach2Drive sample order is whatever the index requested. The audit path
    # already loads left/front/right order; if a caller passes one camera, repeat
    # it to preserve the shape expected by TransFuser.
    if camera.shape[0] == 1:
        camera = np.repeat(camera, 3, axis=0)
    elif camera.shape[0] >= 3:
        camera = camera[:3]
    else:
        camera = np.concatenate([camera, np.zeros_like(camera[:1])], axis=0)[:3]

    stitched = np.concatenate([np.transpose(view, (1, 2, 0)) for view in camera], axis=1)
    rgb = crop_rgb_like_transfuser((stitched * 255.0).clip(0, 255).astype(np.uint8))
    lidar_tf = lidar_bev_to_transfuser(lidar)
    target_point = target_point_from_scalar(scalar)
    velocity = np.asarray([float(scalar[0])], dtype=np.float32)

    return {
        "rgb": torch.from_numpy(rgb).unsqueeze(0),
        "lidar": torch.from_numpy(lidar_tf).unsqueeze(0),
        "target_point": torch.from_numpy(target_point).unsqueeze(0),
        "velocity": torch.from_numpy(velocity).reshape(1, 1),
    }


def load_frame_record(episode_dir: Path, frame_index: int) -> Dict:
    frames_path = episode_dir / "frames.jsonl"
    with frames_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == frame_index:
                return json.loads(line)
    raise IndexError(f"Frame index {frame_index} out of range for {frames_path}")

