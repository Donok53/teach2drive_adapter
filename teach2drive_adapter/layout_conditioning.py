from typing import Dict

import torch
from torch import nn


class SensorLayoutEncoder(nn.Module):
    """Encode camera/LiDAR pose metadata into a compact conditioning vector."""

    def __init__(self, layout_dim: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(layout_dim),
            nn.Linear(layout_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def forward(self, layout: torch.Tensor) -> torch.Tensor:
        return self.net(layout)


class FiLMLayoutAdapter(nn.Module):
    """Feature-wise modulation for a frozen driving backbone feature."""

    def __init__(self, feature_dim: int, layout_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layout_encoder = SensorLayoutEncoder(layout_dim, hidden_dim)
        self.to_scale_shift = nn.Linear(hidden_dim, feature_dim * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, features: torch.Tensor, layout: torch.Tensor) -> Dict[str, torch.Tensor]:
        condition = self.layout_encoder(layout)
        scale, shift = self.to_scale_shift(condition).chunk(2, dim=-1)
        adapted = features * (1.0 + scale) + shift
        return {"features": adapted, "layout_condition": condition}
