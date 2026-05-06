from pathlib import Path
from typing import Dict, Sequence

import torch
from torch import nn

from .layout_conditioning import FiLMLayoutAdapter
from .transfuserpp_bridge import (
    base_target_from_checkpoint,
    load_transfuserpp,
    prepare_transfuserpp_inputs,
    speed_expectation,
)


class TransFuserPPResidualHeads(nn.Module):
    def __init__(self, input_dim: int, target_dim: int, hidden_dim: int = 512, stop_state_classes: int = 4, stop_reason_classes: int = 8) -> None:
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


class TransFuserPPResidualAdapterPolicy(nn.Module):
    """Frozen CARLA Garage TransFuser++ prior plus a small layout-aware residual adapter."""

    def __init__(
        self,
        garage_root: str,
        team_config: str,
        device: torch.device,
        scalar_dim: int,
        target_dim: int,
        layout_dim: int,
        cameras: Sequence[str],
        hidden_dim: int = 512,
        speed_dim: int = 4,
        checkpoint: str = "",
        command_mode: str = "lane_follow",
        tfpp_camera: str = "front",
        layout_hidden_dim: int = 128,
        train_prior_substrings: Sequence[str] = (),
    ) -> None:
        super().__init__()
        self.net, self.config, self.load_info = load_transfuserpp(garage_root=garage_root, team_config=team_config, checkpoint=checkpoint, device=device)
        self.net.eval()
        self.train_prior_substrings = tuple(item for item in train_prior_substrings if item)
        self._configure_prior_train_mode()
        self.scalar_dim = int(scalar_dim)
        self.target_dim = int(target_dim)
        self.layout_dim = int(layout_dim)
        self.speed_dim = int(speed_dim)
        self.cameras = tuple(cameras)
        self.command_mode = command_mode
        self.tfpp_camera = tfpp_camera
        feature_dim = self.scalar_dim + self.layout_dim + self.target_dim + (int(getattr(self.config, "predict_checkpoint_len", 10)) * 2) + len(getattr(self.config, "target_speeds", [])) + 1
        self.layout_adapter = FiLMLayoutAdapter(feature_dim=feature_dim, layout_dim=self.layout_dim, hidden_dim=layout_hidden_dim)
        self.heads = TransFuserPPResidualHeads(feature_dim, self.target_dim, hidden_dim=hidden_dim)

    def _configure_prior_train_mode(self) -> None:
        matched = []
        for name, param in self.net.named_parameters():
            trainable = bool(self.train_prior_substrings) and any(token in name for token in self.train_prior_substrings)
            param.requires_grad = trainable
            if trainable:
                matched.append(name)
        if self.train_prior_substrings and not matched:
            raise ValueError(f"No TransFuser++ parameters matched --train-prior-substrings={list(self.train_prior_substrings)}")
        self.prior_trainable_names = tuple(matched)
        self.prior_trainable = bool(matched)
        self.load_info = {
            **self.load_info,
            "train_prior_substrings": list(self.train_prior_substrings),
            "trainable_prior_tensors": len(self.prior_trainable_names),
            "trainable_prior_tensors_preview": list(self.prior_trainable_names[:20]),
        }

    def prior_trainable_info(self) -> Dict:
        return {
            "enabled": bool(self.prior_trainable),
            "substrings": list(self.train_prior_substrings),
            "tensor_count": len(self.prior_trainable_names),
            "parameter_count": int(sum(param.numel() for param in self.net.parameters() if param.requires_grad)),
            "preview": list(self.prior_trainable_names[:20]),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.eval()
        return self

    def _frozen_prior(self, scalar: torch.Tensor, camera: torch.Tensor, lidar: torch.Tensor):
        inputs = prepare_transfuserpp_inputs(
            scalar=scalar,
            camera=camera,
            lidar=lidar,
            cameras=self.cameras,
            config=self.config,
            command_mode=self.command_mode,
            tfpp_camera=self.tfpp_camera,
        )
        return self.net(**inputs)

    def forward(self, scalar: torch.Tensor, camera: torch.Tensor, lidar: torch.Tensor, layout: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.prior_trainable:
            outputs = self._frozen_prior(scalar, camera, lidar)
            pred_target_speed = outputs[1]
            pred_checkpoint = outputs[2]
            base_target = base_target_from_checkpoint(
                pred_checkpoint=pred_checkpoint,
                pred_target_speed=pred_target_speed,
                scalar=scalar,
                config=self.config,
                target_dim=self.target_dim,
                speed_dim=self.speed_dim,
            )
            checkpoint_flat = torch.zeros((scalar.shape[0], int(getattr(self.config, "predict_checkpoint_len", 10)) * 2), dtype=scalar.dtype, device=scalar.device)
            if pred_checkpoint is not None:
                raw = pred_checkpoint.reshape(pred_checkpoint.shape[0], -1)
                checkpoint_flat[:, : min(raw.shape[1], checkpoint_flat.shape[1])] = raw[:, : checkpoint_flat.shape[1]]
            if pred_target_speed is None:
                speed_logits = torch.zeros((scalar.shape[0], len(getattr(self.config, "target_speeds", []))), dtype=scalar.dtype, device=scalar.device)
            else:
                speed_logits = pred_target_speed
            expected_speed = speed_expectation(pred_target_speed, self.config, scalar.shape[0], scalar.device)
        else:
            with torch.no_grad():
                outputs = self._frozen_prior(scalar, camera, lidar)
                pred_target_speed = outputs[1]
                pred_checkpoint = outputs[2]
                base_target = base_target_from_checkpoint(
                    pred_checkpoint=pred_checkpoint,
                    pred_target_speed=pred_target_speed,
                    scalar=scalar,
                    config=self.config,
                    target_dim=self.target_dim,
                    speed_dim=self.speed_dim,
                ).detach()
                checkpoint_flat = torch.zeros((scalar.shape[0], int(getattr(self.config, "predict_checkpoint_len", 10)) * 2), dtype=scalar.dtype, device=scalar.device)
                if pred_checkpoint is not None:
                    raw = pred_checkpoint.reshape(pred_checkpoint.shape[0], -1)
                    checkpoint_flat[:, : min(raw.shape[1], checkpoint_flat.shape[1])] = raw[:, : checkpoint_flat.shape[1]]
                if pred_target_speed is None:
                    speed_logits = torch.zeros((scalar.shape[0], len(getattr(self.config, "target_speeds", []))), dtype=scalar.dtype, device=scalar.device)
                else:
                    speed_logits = pred_target_speed
                expected_speed = speed_expectation(pred_target_speed, self.config, scalar.shape[0], scalar.device)
        features = torch.cat([scalar, layout, base_target, checkpoint_flat, speed_logits, expected_speed], dim=1)
        adapted = self.layout_adapter(features, layout)["features"]
        return self.heads(adapted, base_target)

    def extra_state(self) -> Dict:
        return {"transfuserpp_load_info": self.load_info}

