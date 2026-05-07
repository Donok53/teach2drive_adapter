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


def camera_to_transfuserpp_rgb(
    camera: torch.Tensor,
    cameras: Sequence[str],
    config,
    tfpp_camera: str = "front",
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
    if bool(getattr(config, "crop_image", True)):
        crop_h = int(getattr(config, "cropped_height", 384))
        crop_w = int(getattr(config, "cropped_width", 1024))
        side_crop = max((rgb.shape[-1] - crop_w) // 2, 0)
        rgb = rgb[:, :, :crop_h, side_crop : side_crop + crop_w]
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
) -> Dict[str, torch.Tensor]:
    target_point = target_point_from_scalar(scalar)
    return {
        "rgb": camera_to_transfuserpp_rgb(camera, cameras=cameras, config=config, tfpp_camera=tfpp_camera),
        "lidar_bev": lidar_to_transfuserpp_bev(lidar, config=config),
        "target_point": target_point,
        "ego_vel": scalar[:, :1].contiguous(),
        "command": command_from_target_point(target_point, mode=command_mode),
    }
