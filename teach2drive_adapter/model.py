from typing import Dict

import torch
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PortableBackbone(nn.Module):
    """Sensor backbone intended to be replaced by stronger pretrained policies."""

    def __init__(self, scalar_dim: int, num_cameras: int, embed_dim: int = 256, lidar_channels: int = 1) -> None:
        super().__init__()
        self.num_cameras = int(num_cameras)
        self.image_encoder = ConvEncoder(3, embed_dim)
        self.lidar_encoder = ConvEncoder(lidar_channels, embed_dim)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.camera_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.lidar_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scalar_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 4,
                dropout=0.05,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, scalar: torch.Tensor, camera: torch.Tensor, lidar: torch.Tensor) -> torch.Tensor:
        batch, cameras, channels, height, width = camera.shape
        image = camera.reshape(batch * cameras, channels, height, width)
        image_feat = self.image_encoder(image).reshape(batch, cameras, -1)
        lidar_feat = self.lidar_encoder(lidar).unsqueeze(1)
        scalar_feat = self.scalar_encoder(scalar).unsqueeze(1)
        tokens = torch.cat(
            [
                image_feat + self.camera_token,
                lidar_feat + self.lidar_token,
                scalar_feat + self.scalar_token,
            ],
            dim=1,
        )
        fused = self.fusion(tokens)
        return self.norm(fused.mean(dim=1))


class BottleneckAdapter(nn.Module):
    def __init__(self, embed_dim: int = 256, bottleneck_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PolicyHeads(nn.Module):
    def __init__(self, embed_dim: int, target_dim: int, stop_state_classes: int = 4, stop_reason_classes: int = 8) -> None:
        super().__init__()
        self.trajectory = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, target_dim))
        self.stop_state = nn.Linear(embed_dim, stop_state_classes)
        self.stop_reason = nn.Linear(embed_dim, stop_reason_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "target": self.trajectory(x),
            "stop_state": self.stop_state(x),
            "stop_reason": self.stop_reason(x),
        }


class Teach2DriveAdapterPolicy(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        num_cameras: int,
        target_dim: int,
        embed_dim: int = 256,
        adapter_dim: int = 64,
        lidar_channels: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = PortableBackbone(scalar_dim, num_cameras, embed_dim=embed_dim, lidar_channels=lidar_channels)
        self.adapter = BottleneckAdapter(embed_dim=embed_dim, bottleneck_dim=adapter_dim)
        self.heads = PolicyHeads(embed_dim=embed_dim, target_dim=target_dim)

    def forward(self, scalar: torch.Tensor, camera: torch.Tensor, lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(scalar, camera, lidar)
        adapted = self.adapter(features)
        return self.heads(adapted)


def configure_train_mode(model: Teach2DriveAdapterPolicy, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad = True
    if mode == "scratch" or mode == "full":
        return
    if mode == "adapter":
        for param in model.backbone.parameters():
            param.requires_grad = False
        return
    if mode == "head":
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.adapter.parameters():
            param.requires_grad = False
        return
    raise ValueError(f"Unknown training mode: {mode}")


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}

