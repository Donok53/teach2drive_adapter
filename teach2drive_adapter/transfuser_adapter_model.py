from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .eval_transfuser_openloop import _load_net, _target_point_image


class ResidualPolicyHeads(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_dim: int = 512,
        stop_state_classes: int = 4,
        stop_reason_classes: int = 8,
    ) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )
        self.stop_state = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, stop_state_classes))
        self.stop_reason = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, stop_reason_classes))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features: torch.Tensor, base_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "target": base_target + self.residual(features),
            "stop_state": self.stop_state(features),
            "stop_reason": self.stop_reason(features),
        }


class TransFuserResidualAdapterPolicy(nn.Module):
    """Frozen official TransFuser prior plus a small Teach2Drive residual head."""

    def __init__(
        self,
        transfuser_root: str,
        team_config: str,
        device: torch.device,
        scalar_dim: int,
        target_dim: int,
        hidden_dim: int = 512,
        speed_dim: int = 4,
        image_hw: Tuple[int, int] = (160, 704),
        lidar_hw: Tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        self.net, self.config, self.draw_target_point, self.load_info = _load_net(Path(transfuser_root).expanduser(), Path(team_config).expanduser(), device)
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
        self.scalar_dim = int(scalar_dim)
        self.target_dim = int(target_dim)
        self.speed_dim = int(speed_dim)
        self.traj_dim = self.target_dim - self.speed_dim - 1
        if self.traj_dim <= 0 or self.traj_dim % 3 != 0:
            raise ValueError(f"Expected target_dim = 3*horizons + speed_dim + stop, got {self.target_dim}")
        self.horizon_count = self.traj_dim // 3
        self.image_hw = tuple(int(v) for v in image_hw)
        self.lidar_hw = tuple(int(v) for v in lidar_hw)
        feature_dim = 512 + self.scalar_dim + self.target_dim
        self.heads = ResidualPolicyHeads(feature_dim, self.target_dim, hidden_dim=hidden_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.eval()
        return self

    def _camera_to_transfuser_rgb(self, camera: torch.Tensor) -> torch.Tensor:
        batch, num_cameras, channels, height, width = camera.shape
        if num_cameras == 1:
            camera = camera.repeat(1, 3, 1, 1, 1)
        elif num_cameras < 3:
            pad = torch.zeros((batch, 3 - num_cameras, channels, height, width), dtype=camera.dtype, device=camera.device)
            camera = torch.cat([camera, pad], dim=1)
        else:
            camera = camera[:, :3]

        rgb = camera * 255.0
        stitched = torch.cat([rgb[:, idx] for idx in range(3)], dim=-1)
        crop_h, crop_w = self.image_hw
        if stitched.shape[-2] < crop_h or stitched.shape[-1] < crop_w:
            target_h = max(int(stitched.shape[-2]), crop_h)
            target_w = max(int(stitched.shape[-1]), crop_w)
            stitched = F.interpolate(stitched, size=(target_h, target_w), mode="bilinear", align_corners=False)
        start_y = max((int(stitched.shape[-2]) - crop_h) // 2, 0)
        start_x = max((int(stitched.shape[-1]) - crop_w) // 2, 0)
        return stitched[:, :, start_y : start_y + crop_h, start_x : start_x + crop_w].contiguous()

    def _lidar_to_transfuser_bev(self, lidar: torch.Tensor) -> torch.Tensor:
        if lidar.shape[1] == 1:
            lidar = torch.cat([lidar, torch.zeros_like(lidar)], dim=1)
        elif lidar.shape[1] > 2:
            lidar = lidar[:, :2]
        if tuple(lidar.shape[-2:]) != self.lidar_hw:
            lidar = F.interpolate(lidar, size=self.lidar_hw, mode="bilinear", align_corners=False)
        return lidar.contiguous()

    def _target_point_from_scalar(self, scalar: torch.Tensor) -> torch.Tensor:
        if scalar.shape[1] < 12:
            return torch.zeros((scalar.shape[0], 2), dtype=scalar.dtype, device=scalar.device)
        return scalar[:, 10:12].contiguous()

    def _target_point_image(self, target_point: torch.Tensor) -> torch.Tensor:
        return _target_point_image(self.draw_target_point, target_point, target_point.device)

    def _base_target_from_transfuser(self, pred_wp_lidar: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        pred_wp_vehicle = pred_wp_lidar.clone()
        pred_wp_vehicle[:, :, 0] += float(self.config.lidar_pos[0])
        base = torch.zeros((scalar.shape[0], self.target_dim), dtype=scalar.dtype, device=scalar.device)
        usable_horizons = min(self.horizon_count, pred_wp_vehicle.shape[1])
        for idx in range(usable_horizons):
            base[:, idx * 3 : idx * 3 + 2] = pred_wp_vehicle[:, idx, :2]
        if self.speed_dim > 0:
            velocity = scalar[:, :1].repeat(1, self.speed_dim)
            base[:, self.traj_dim : self.traj_dim + self.speed_dim] = velocity
        return base

    def _frozen_transfuser(self, rgb: torch.Tensor, lidar: torch.Tensor, target_point: torch.Tensor, velocity: torch.Tensor):
        target_point_image = self._target_point_image(target_point)
        lidar_input = torch.cat((lidar, target_point_image), dim=1) if self.config.use_target_point_image else lidar
        features, _image_features_grid, fused_features = self.net._model(rgb, lidar_input, velocity)
        pred_wp_lidar, _, _, _, _ = self.net.forward_gru(fused_features, target_point)
        return fused_features, pred_wp_lidar

    def forward(self, scalar: torch.Tensor, camera: torch.Tensor, lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb = self._camera_to_transfuser_rgb(camera)
        lidar_bev = self._lidar_to_transfuser_bev(lidar)
        target_point = self._target_point_from_scalar(scalar)
        velocity = scalar[:, :1].contiguous()
        with torch.no_grad():
            fused_features, pred_wp_lidar = self._frozen_transfuser(rgb, lidar_bev, target_point, velocity)
            fused_features = fused_features.detach()
            base_target = self._base_target_from_transfuser(pred_wp_lidar, scalar).detach()
        adapter_features = torch.cat([fused_features, scalar, base_target], dim=1)
        return self.heads(adapter_features, base_target)

    def extra_state(self) -> Dict:
        return {"transfuser_load_info": self.load_info}
