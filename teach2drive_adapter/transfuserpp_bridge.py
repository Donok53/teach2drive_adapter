from __future__ import annotations

import contextlib
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


_WARP_GRID_CACHE: dict[tuple, torch.Tensor] = {}


@dataclass(frozen=True)
class TransFuserPPBatchSpec:
    image_hw: Tuple[int, int] = (384, 1024)
    lidar_hw: Tuple[int, int] = (256, 256)
    command_mode: str = "lane_follow"
    tfpp_camera: str = "front"


def _install_runtime_stubs() -> None:
    """Install tiny import stubs for CARLA-only modules not used in offline training."""

    if not getattr(nn.TransformerDecoderLayer, "_teach2drive_activation_compat", False):
        original_decoder_layer = nn.TransformerDecoderLayer

        def decoder_layer_activation_compat(*args, **kwargs):
            activation = kwargs.get("activation")
            if isinstance(activation, nn.GELU):
                kwargs["activation"] = "gelu"
            elif isinstance(activation, nn.ReLU):
                kwargs["activation"] = "relu"
            return original_decoder_layer(*args, **kwargs)

        decoder_layer_activation_compat._teach2drive_activation_compat = True  # type: ignore[attr-defined]
        nn.TransformerDecoderLayer = decoder_layer_activation_compat  # type: ignore[assignment]

    if not getattr(nn.CrossEntropyLoss, "_teach2drive_label_smoothing_compat", False):
        original_ce_loss = nn.CrossEntropyLoss

        class CrossEntropyLossCompat(original_ce_loss):  # type: ignore[misc, valid-type]
            _teach2drive_label_smoothing_compat = True

            def __init__(self, *args, **kwargs):
                kwargs.pop("label_smoothing", None)
                super().__init__(*args, **kwargs)

        nn.CrossEntropyLoss = CrossEntropyLossCompat  # type: ignore[assignment]

    try:
        torch.meshgrid(torch.arange(1), torch.arange(1), indexing="ij")
    except TypeError:
        original_meshgrid = torch.meshgrid

        def meshgrid_compat(*tensors, indexing=None):
            return original_meshgrid(*tensors)

        torch.meshgrid = meshgrid_compat  # type: ignore[assignment]

    if "carla" not in sys.modules:
        carla = types.ModuleType("carla")

        class _Vector3D:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = x
                self.y = y
                self.z = z

            def __add__(self, other):
                return self.__class__(self.x + other.x, self.y + other.y, self.z + other.z)

        class _Location(_Vector3D):
            pass

        class _Color:
            def __init__(self, r=0, g=0, b=0, a=255):
                self.r = r
                self.g = g
                self.b = b
                self.a = a

        carla.Vector3D = _Vector3D
        carla.Location = _Location
        carla.Color = _Color
        sys.modules["carla"] = carla

    if "agents.navigation.global_route_planner" not in sys.modules:
        agents = sys.modules.setdefault("agents", types.ModuleType("agents"))
        navigation = sys.modules.setdefault("agents.navigation", types.ModuleType("agents.navigation"))
        grp = types.ModuleType("agents.navigation.global_route_planner")

        class GlobalRoutePlanner:  # pragma: no cover - offline import shim only
            pass

        grp.GlobalRoutePlanner = GlobalRoutePlanner
        agents.navigation = navigation
        navigation.global_route_planner = grp
        sys.modules["agents.navigation.global_route_planner"] = grp


@contextlib.contextmanager
def _force_timm_no_pretrained():
    import timm

    original = timm.create_model

    def create_model_no_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        return original(*args, **kwargs)

    timm.create_model = create_model_no_pretrained
    try:
        yield
    finally:
        timm.create_model = original


def _insert_team_code(garage_root: Path) -> None:
    team_code = garage_root / "team_code"
    if not team_code.exists():
        raise FileNotFoundError(f"Could not find CARLA Garage team_code at {team_code}")
    team_code_str = str(team_code)
    if team_code_str not in sys.path:
        sys.path.insert(0, team_code_str)


def _decode_jsonpickle_value(value):
    if isinstance(value, Mapping):
        if "py/tuple" in value:
            return tuple(_decode_jsonpickle_value(item) for item in value["py/tuple"])
        if value.get("py/object") == "numpy.ndarray" and "values" in value:
            return _decode_jsonpickle_value(value["values"])
        return {key: _decode_jsonpickle_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_jsonpickle_value(item) for item in value]
    return value


def _load_config(team_config: Path, garage_root: Path):
    _install_runtime_stubs()
    _insert_team_code(garage_root)
    from config import GlobalConfig  # type: ignore

    config = GlobalConfig()
    config_path = team_config / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing TransFuser++ config.json: {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    config.__dict__.update({key: _decode_jsonpickle_value(value) for key, value in raw_config.items()})
    return config


def _load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, Mapping) and "model_state" in state:
        state = state["model_state"]
    if all(str(key).startswith("module.") for key in state.keys()):
        state = {str(key)[7:]: value for key, value in state.items()}
    return dict(state)


def load_transfuserpp(
    garage_root: str,
    team_config: str,
    device: torch.device,
    checkpoint: str = "",
) -> Tuple[nn.Module, object, Dict]:
    """Load CARLA Garage TransFuser++ with official checkpoint weights."""

    garage_root_path = Path(garage_root).expanduser().resolve()
    team_config_path = Path(team_config).expanduser().resolve()
    config = _load_config(team_config_path, garage_root_path)
    _insert_team_code(garage_root_path)
    from model import LidarCenterNet  # type: ignore

    with _force_timm_no_pretrained():
        net = LidarCenterNet(config)

    checkpoint_path = Path(checkpoint).expanduser() if checkpoint else sorted(team_config_path.glob("model_*.pth"))[0]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing TransFuser++ checkpoint: {checkpoint_path}")
    state = _load_state_dict(checkpoint_path, device)
    missing, unexpected = net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net, config, {
        "garage_root": str(garage_root_path),
        "team_config": str(team_config_path),
        "checkpoint": str(checkpoint_path),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def _camera_index(cameras: Sequence[str], target: str) -> int:
    if target in cameras:
        return int(cameras.index(target))
    if "front" in cameras:
        return int(cameras.index("front"))
    return 0


def _pose6(value: Optional[Sequence[float]], default: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    raw = list(default if value is None else value)
    if len(raw) < 3:
        raise ValueError(f"Camera pose must contain at least x,y,z, got {raw}")
    while len(raw) < 6:
        raw.append(0.0)
    return tuple(float(v) for v in raw[:6])  # type: ignore[return-value]


def _ground_to_image_homography(
    width: int,
    height: int,
    fov_deg: float,
    pose: Sequence[float],
    ground_z_m: float = 0.0,
) -> torch.Tensor:
    """Return a ground-plane-to-image homography for the front camera.

    The camera and ego frames use x forward, image x right, and image y down.
    This is exact for points on the configured ground plane. Non-ground objects
    are still a single-view approximation because RGB lacks depth.
    """

    x, y, z, _roll, _pitch, yaw = _pose6(pose, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    focal = float(width) / (2.0 * math.tan(math.radians(float(fov_deg)) / 2.0))
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    k = torch.tensor(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    # Vehicle ground point P=[X,Y,1] at Z=ground_z.  For yaw=0:
    # camera_x = Y - camera_y, camera_y_down = camera_z - Z, camera_z = X - camera_x.
    yaw_rad = math.radians(float(yaw))
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    ego_to_cam = torch.tensor(
        [
            [-sin_y, cos_y, sin_y * x - cos_y * y],
            [0.0, 0.0, z - float(ground_z_m)],
            [cos_y, sin_y, -cos_y * x - sin_y * y],
        ],
        dtype=torch.float64,
    )
    return k @ ego_to_cam


def _ground_plane_warp_grid(
    *,
    height: int,
    width: int,
    fov_deg: float,
    source_pose: Sequence[float],
    target_pose: Sequence[float],
    ground_z_m: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    source_pose6 = _pose6(source_pose, (1.25, 0.0, 1.95, 0.0, 0.0, 0.0))
    target_pose6 = _pose6(target_pose, (-1.5, 0.0, 2.0, 0.0, 0.0, 0.0))
    key = (
        str(device),
        str(dtype),
        int(height),
        int(width),
        round(float(fov_deg), 6),
        tuple(round(v, 6) for v in source_pose6),
        tuple(round(v, 6) for v in target_pose6),
        round(float(ground_z_m), 6),
    )
    cached = _WARP_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    h_source = _ground_to_image_homography(width, height, fov_deg, source_pose6, ground_z_m)
    h_target = _ground_to_image_homography(width, height, fov_deg, target_pose6, ground_z_m)
    target_to_source = h_source @ torch.linalg.inv(h_target)
    target_to_source = target_to_source.to(device=device, dtype=dtype)

    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    flat = torch.stack((xs, ys, ones), dim=0).reshape(3, -1)
    src = target_to_source @ flat
    denom = src[2].clamp(min=1e-6)
    x_src = (src[0] / denom).reshape(height, width)
    y_src = (src[1] / denom).reshape(height, width)
    x_norm = 2.0 * x_src / max(width - 1, 1) - 1.0
    y_norm = 2.0 * y_src / max(height - 1, 1) - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0).contiguous()
    _WARP_GRID_CACHE[key] = grid
    return grid


def ground_plane_warp_rgb(
    rgb: torch.Tensor,
    *,
    enabled: bool,
    fov_deg: float,
    source_pose: Sequence[float],
    target_pose: Sequence[float],
    ground_z_m: float = 0.0,
) -> torch.Tensor:
    if not enabled:
        return rgb
    if rgb.ndim != 4:
        raise ValueError(f"Expected RGB tensor [B,3,H,W], got {tuple(rgb.shape)}")
    height = int(rgb.shape[-2])
    width = int(rgb.shape[-1])
    grid = _ground_plane_warp_grid(
        height=height,
        width=width,
        fov_deg=float(fov_deg),
        source_pose=source_pose,
        target_pose=target_pose,
        ground_z_m=float(ground_z_m),
        device=rgb.device,
        dtype=rgb.dtype,
    )
    return F.grid_sample(
        rgb,
        grid.expand(rgb.shape[0], -1, -1, -1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).contiguous()


def camera_to_transfuserpp_rgb(
    camera: torch.Tensor,
    cameras: Sequence[str],
    config,
    tfpp_camera: str = "front",
    crop_shift_x_px: float = 0.0,
    crop_shift_y_px: float = 0.0,
    crop_scale: float = 1.0,
    ground_plane_warp: bool = False,
    ground_plane_source_pose: Optional[Sequence[float]] = None,
    ground_plane_target_pose: Optional[Sequence[float]] = None,
    ground_plane_z_m: float = 0.0,
) -> torch.Tensor:
    """Convert Teach2Drive camera tensors in [0, 1] to TF++ cropped RGB in [0, 255]."""

    if camera.ndim != 5:
        raise ValueError(f"Expected camera tensor [B,N,3,H,W], got {tuple(camera.shape)}")
    index = _camera_index(cameras, tfpp_camera)
    rgb = camera[:, index] * 255.0
    resize_h = int(getattr(config, "camera_height", 512))
    resize_w = int(getattr(config, "camera_width", 1024))
    if tuple(rgb.shape[-2:]) != (resize_h, resize_w):
        rgb = F.interpolate(rgb, size=(resize_h, resize_w), mode="bilinear", align_corners=False)
    rgb = ground_plane_warp_rgb(
        rgb,
        enabled=bool(ground_plane_warp),
        fov_deg=float(getattr(config, "camera_fov", getattr(config, "fov", 110.0))),
        source_pose=_pose6(ground_plane_source_pose, (1.25, 0.0, 1.95, 0.0, 0.0, 0.0)),
        target_pose=_pose6(ground_plane_target_pose, (-1.5, 0.0, 2.0, 0.0, 0.0, 0.0)),
        ground_z_m=float(ground_plane_z_m),
    )
    scale = float(crop_scale)
    if abs(scale - 1.0) > 1e-6:
        scaled_h = max(1, int(round(resize_h * scale)))
        scaled_w = max(1, int(round(resize_w * scale)))
        scaled = F.interpolate(rgb, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)
        pad_y_total = max(resize_h - scaled_h, 0)
        pad_x_total = max(resize_w - scaled_w, 0)
        if pad_y_total or pad_x_total:
            scaled = F.pad(
                scaled,
                (
                    pad_x_total // 2,
                    pad_x_total - pad_x_total // 2,
                    pad_y_total // 2,
                    pad_y_total - pad_y_total // 2,
                ),
            )
        if scaled.shape[-2] > resize_h or scaled.shape[-1] > resize_w:
            start_y = max((scaled.shape[-2] - resize_h) // 2, 0)
            start_x = max((scaled.shape[-1] - resize_w) // 2, 0)
            scaled = scaled[:, :, start_y : start_y + resize_h, start_x : start_x + resize_w]
        rgb = scaled
    if bool(getattr(config, "crop_image", True)):
        crop_h = int(getattr(config, "cropped_height", 384))
        crop_w = int(getattr(config, "cropped_width", 1024))
        padded_h = max(int(rgb.shape[-2]), crop_h + abs(int(round(float(crop_shift_y_px)))) * 2)
        padded_w = max(int(rgb.shape[-1]), crop_w + abs(int(round(float(crop_shift_x_px)))) * 2)
        pad_top = max((padded_h - int(rgb.shape[-2])) // 2, 0)
        pad_bottom = max(padded_h - int(rgb.shape[-2]) - pad_top, 0)
        pad_left = max((padded_w - int(rgb.shape[-1])) // 2, 0)
        pad_right = max(padded_w - int(rgb.shape[-1]) - pad_left, 0)
        if pad_top or pad_bottom or pad_left or pad_right:
            rgb = F.pad(rgb, (pad_left, pad_right, pad_top, pad_bottom))
        side_crop = max((rgb.shape[-1] - crop_w) // 2, 0)
        start_y = pad_top + int(round(float(crop_shift_y_px)))
        start_x = side_crop + int(round(float(crop_shift_x_px)))
        start_y = max(0, min(start_y, int(rgb.shape[-2]) - crop_h))
        start_x = max(0, min(start_x, int(rgb.shape[-1]) - crop_w))
        rgb = rgb[:, :, start_y : start_y + crop_h, start_x : start_x + crop_w]
    return rgb.contiguous()


def lidar_to_transfuserpp_bev(lidar: torch.Tensor, config) -> torch.Tensor:
    """Map Teach2Drive BEV tensors to the TF++ LiDAR histogram shape."""

    channels = int(getattr(config, "lidar_seq_len", 1)) * (1 + int(getattr(config, "use_ground_plane", 0)))
    if lidar.shape[1] < channels:
        pad = torch.zeros(
            (lidar.shape[0], channels - lidar.shape[1], lidar.shape[2], lidar.shape[3]),
            dtype=lidar.dtype,
            device=lidar.device,
        )
        lidar = torch.cat([lidar, pad], dim=1)
    elif lidar.shape[1] > channels:
        lidar = lidar[:, :channels]
    lidar_h = int(getattr(config, "lidar_resolution_height", 256))
    lidar_w = int(getattr(config, "lidar_resolution_width", 256))
    if tuple(lidar.shape[-2:]) != (lidar_h, lidar_w):
        lidar = F.interpolate(lidar, size=(lidar_h, lidar_w), mode="bilinear", align_corners=False)
    return lidar.contiguous()


def translate_lidar_bev_meters(
    lidar: torch.Tensor,
    shift_x_m: float = 0.0,
    shift_y_m: float = 0.0,
    pixels_per_meter: float = 4.0,
) -> torch.Tensor:
    """Translate an ego-frame BEV histogram by a metric offset before TF++ sees it.

    The collected BEV is indexed as [B, C, y, x]. Positive x moves occupancy
    forward in the vehicle frame; positive y moves it left in the vehicle frame.
    """

    shift_x_px = float(shift_x_m) * float(pixels_per_meter)
    shift_y_px = float(shift_y_m) * float(pixels_per_meter)
    if abs(shift_x_px) < 1e-6 and abs(shift_y_px) < 1e-6:
        return lidar
    if lidar.ndim != 4:
        raise ValueError(f"Expected LiDAR BEV [B,C,H,W], got {tuple(lidar.shape)}")
    height = int(lidar.shape[-2])
    width = int(lidar.shape[-1])
    y_base, x_base = torch.meshgrid(
        torch.arange(height, dtype=lidar.dtype, device=lidar.device),
        torch.arange(width, dtype=lidar.dtype, device=lidar.device),
        indexing="ij",
    )
    x_src = x_base - shift_x_px
    y_src = y_base - shift_y_px
    x_norm = 2.0 * x_src / max(width - 1, 1) - 1.0
    y_norm = 2.0 * y_src / max(height - 1, 1) - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0).expand(lidar.shape[0], -1, -1, -1)
    return F.grid_sample(lidar, grid, mode="bilinear", padding_mode="zeros", align_corners=True).contiguous()


def target_point_from_scalar(scalar: torch.Tensor) -> torch.Tensor:
    if scalar.shape[1] < 12:
        return torch.zeros((scalar.shape[0], 2), dtype=scalar.dtype, device=scalar.device)
    return scalar[:, 10:12].contiguous()


def command_from_target_point(target_point: torch.Tensor, mode: str = "lane_follow") -> torch.Tensor:
    command = torch.zeros((target_point.shape[0], 6), dtype=target_point.dtype, device=target_point.device)
    if mode == "target_angle":
        angle = torch.atan2(target_point[:, 1], torch.clamp(target_point[:, 0], min=1e-3))
        # CARLA commands are one-hot after subtracting 1:
        # 0 left, 1 right, 2 straight, 3 lane follow, 4/5 lane changes.
        indices = torch.full((target_point.shape[0],), 2, dtype=torch.long, device=target_point.device)
        indices = torch.where(angle > 0.35, torch.zeros_like(indices), indices)
        indices = torch.where(angle < -0.35, torch.ones_like(indices), indices)
        command.scatter_(1, indices[:, None], 1.0)
        return command
    if mode != "lane_follow":
        raise ValueError(f"Unknown command mode: {mode}")
    command[:, 3] = 1.0
    return command


def speed_expectation(pred_target_speed: Optional[torch.Tensor], config, batch_size: int, device: torch.device) -> torch.Tensor:
    if pred_target_speed is None:
        return torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    target_speeds = torch.tensor(getattr(config, "target_speeds", [0.0]), dtype=pred_target_speed.dtype, device=device)
    probs = torch.softmax(pred_target_speed, dim=1)
    return torch.sum(probs * target_speeds[None, :], dim=1, keepdim=True)


def xy_to_yaw(xy: torch.Tensor) -> torch.Tensor:
    prev = torch.cat([torch.zeros_like(xy[:, :1]), xy[:, :-1]], dim=1)
    delta = xy - prev
    return torch.atan2(delta[..., 1], torch.clamp(delta[..., 0], min=1e-3))


def base_target_from_checkpoint(
    pred_checkpoint: Optional[torch.Tensor],
    pred_target_speed: Optional[torch.Tensor],
    scalar: torch.Tensor,
    config,
    target_dim: int,
    speed_dim: int,
) -> torch.Tensor:
    traj_dim = int(target_dim) - int(speed_dim) - 1
    horizon_count = max(traj_dim // 3, 0)
    base = torch.zeros((scalar.shape[0], target_dim), dtype=scalar.dtype, device=scalar.device)
    if pred_checkpoint is not None and horizon_count:
        checkpoints = pred_checkpoint[:, :horizon_count, :2]
        usable = checkpoints.shape[1]
        yaws = xy_to_yaw(checkpoints)
        for idx in range(usable):
            base[:, idx * 3 : idx * 3 + 2] = checkpoints[:, idx, :2]
            base[:, idx * 3 + 2] = yaws[:, idx]
    expected_speed = speed_expectation(pred_target_speed, config, scalar.shape[0], scalar.device)
    if speed_dim > 0:
        base[:, traj_dim : traj_dim + speed_dim] = expected_speed.repeat(1, speed_dim)
    base[:, -1] = (0.5 - expected_speed[:, 0]) * 2.0
    return base


def prepare_transfuserpp_inputs(
    scalar: torch.Tensor,
    camera: torch.Tensor,
    lidar: torch.Tensor,
    cameras: Sequence[str],
    config,
    command_mode: str = "lane_follow",
    tfpp_camera: str = "front",
    camera_crop_shift_x_px: float = 0.0,
    camera_crop_shift_y_px: float = 0.0,
    camera_crop_scale: float = 1.0,
    camera_ground_plane_warp: bool = False,
    camera_ground_plane_source_pose: Optional[Sequence[float]] = None,
    camera_ground_plane_target_pose: Optional[Sequence[float]] = None,
    camera_ground_plane_z_m: float = 0.0,
    lidar_shift_x_m: float = 0.0,
    lidar_shift_y_m: float = 0.0,
    lidar_pixels_per_meter: float = 4.0,
) -> Dict[str, torch.Tensor]:
    target_point = target_point_from_scalar(scalar)
    lidar_bev = lidar_to_transfuserpp_bev(lidar, config=config)
    lidar_bev = translate_lidar_bev_meters(
        lidar_bev,
        shift_x_m=float(lidar_shift_x_m),
        shift_y_m=float(lidar_shift_y_m),
        pixels_per_meter=float(lidar_pixels_per_meter),
    )
    return {
        "rgb": camera_to_transfuserpp_rgb(
            camera,
            cameras=cameras,
            config=config,
            tfpp_camera=tfpp_camera,
            crop_shift_x_px=float(camera_crop_shift_x_px),
            crop_shift_y_px=float(camera_crop_shift_y_px),
            crop_scale=float(camera_crop_scale),
            ground_plane_warp=bool(camera_ground_plane_warp),
            ground_plane_source_pose=camera_ground_plane_source_pose,
            ground_plane_target_pose=camera_ground_plane_target_pose,
            ground_plane_z_m=float(camera_ground_plane_z_m),
        ),
        "lidar_bev": lidar_bev,
        "target_point": target_point,
        "ego_vel": scalar[:, :1].contiguous(),
        "command": command_from_target_point(target_point, mode=command_mode),
    }
