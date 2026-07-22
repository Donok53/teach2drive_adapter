from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .cache_transfuserpp_feature_fusion_features import (
    ALL_FEATURE_NAMES,
    FUSED_FEATURE_NAME,
    _capture_backbone_feature_fusion_features,
)
from .data import STOP_REASON_NAMES, Teach2DriveIndexDataset, split_by_episode
from .peft_lora import (
    install_lora_adapters,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    set_lora_train_mode,
)
from .train_adapter import _per_sample_vector, _weighted_mean
from .train_transfuserpp_feature_then_fusion_adapter import (
    ExtrinsicAwareFeatureThenFusionAdapter,
    FeatureThenFusionAdapter,
    build_extrinsic_vector,
    load_feature_then_fusion_checkpoint,
)
from .train_transfuserpp_fused_feature_policy_adapter import (
    _launch_mask,
    _moving_mask,
    _release_mask,
    _target_speed_mask,
)
from .transfuserpp_bridge import base_target_from_checkpoint, load_transfuserpp, prepare_transfuserpp_inputs


def _camera_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_stage_adapter_layers(raw: str | None) -> tuple[int, ...] | None:
    value = (raw or "all").strip().lower()
    if value in {"", "all", "*"}:
        return None
    if value.startswith("early:"):
        count = int(value.split(":", 1)[1])
        return tuple(range(max(0, count)))
    layers = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        layers.append(int(item))
    return tuple(sorted(set(layers)))


def _split_patterns(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _matches_any(name: str, patterns: list[str]) -> bool:
    return bool(patterns) and any(re.search(pattern, name) for pattern in patterns)


def _unfreeze_matching_tfpp_parameters(net: nn.Module, include: str, exclude: str) -> list[str]:
    include_patterns = _split_patterns(include)
    exclude_patterns = _split_patterns(exclude)
    if not include_patterns:
        return []
    unfrozen = []
    for name, param in net.named_parameters():
        if not _matches_any(name, include_patterns):
            continue
        if _matches_any(name, exclude_patterns):
            continue
        param.requires_grad_(True)
        unfrozen.append(name)
    if not unfrozen:
        raise ValueError(f"No TF++ parameters matched --unfreeze-include={include!r} --unfreeze-exclude={exclude!r}")
    return unfrozen


def _hazard_reason_mask(stop_reason: torch.Tensor, names: str) -> torch.Tensor:
    ids = []
    for name in names.split(","):
        name = name.strip()
        if name and name in STOP_REASON_NAMES:
            ids.append(STOP_REASON_NAMES.index(name))
    if not ids:
        return torch.zeros_like(stop_reason, dtype=torch.bool)
    mask = torch.zeros_like(stop_reason, dtype=torch.bool)
    for idx in ids:
        mask = mask | (stop_reason == int(idx))
    return mask.reshape(-1).bool()


def _recovery_perturb_config() -> dict:
    """ChauffeurNet-style yaw recovery augmentation (view-synthesis-free).
    Enabled via env: T2D_PERTURB_PROB (0..1), T2D_PERTURB_PSI_MAX_DEG, T2D_PERTURB_FOCAL_PX."""
    prob = float(os.environ.get("T2D_PERTURB_PROB", "0") or 0.0)
    psi_max = float(os.environ.get("T2D_PERTURB_PSI_MAX_DEG", "6") or 6.0) * 3.141592653589793 / 180.0
    focal = float(os.environ.get("T2D_PERTURB_FOCAL_PX", "224") or 224.0)
    return {"prob": prob, "psi": psi_max, "focal": focal}


def _xmodal_config() -> dict:
    """Cross-modal (camera<->lidar) alignment loss. Re-anchors the shifted camera
    to the (near-stationary) lidar via a per-frame InfoNCE consistency between the
    adapter's camera and lidar stage features. Env: T2D_XMODAL_WEIGHT, T2D_XMODAL_TEMP."""
    w = float(os.environ.get("T2D_XMODAL_WEIGHT", "0") or 0.0)
    t = float(os.environ.get("T2D_XMODAL_TEMP", "0.1") or 0.1)
    return {"weight": w, "temp": t}


def _apply_recovery_perturbation(camera, scalar, target, cfg):
    """Pan the front camera (crop-shift approximates a yaw) and re-express the
    expert trajectory + navigation target_point in the perturbed ego frame, so
    the model learns to steer back onto the expert path from an off-heading
    state. LiDAR is left unchanged (small yaw-only inconsistency; camera drives
    steering). Sign convention is verified against the pretrained model before use."""
    B = camera.shape[0]
    dev = camera.device
    do = (torch.rand(B, device=dev) < float(cfg["prob"]))
    psi = (torch.rand(B, device=dev) * 2.0 - 1.0) * float(cfg["psi"]) * do.to(camera.dtype)
    # camera pan: +psi (yaw left) makes scene appear shifted right -> roll +du (content toward +width)
    du = torch.round(-float(cfg["focal"]) * psi).to(torch.long)  # sign verified against pretrained model oracle
    cam = camera.clone()
    for b in range(B):
        s = int(du[b].item())
        if s == 0:
            continue
        cam[b] = torch.roll(camera[b], shifts=s, dims=-1)
        if s > 0:
            cam[b, ..., :s] = 0.0
        else:
            cam[b, ..., s:] = 0.0
    c = torch.cos(psi)
    s2 = torch.sin(psi)
    # rotate a set of (x,y) into the perturbed ego frame via R(-psi): x'=x c + y s2 ; y'=-x s2 + y c
    sc = scalar.clone()
    tx = sc[:, 10].clone()
    ty = sc[:, 11].clone()
    sc[:, 10] = tx * c + ty * s2
    sc[:, 11] = -tx * s2 + ty * c
    tgt = target.clone()
    pts = tgt[:, :12].reshape(B, 4, 3).clone()
    px = pts[:, :, 0].clone()
    py = pts[:, :, 1].clone()
    c1 = c.unsqueeze(1)
    s1 = s2.unsqueeze(1)
    pts[:, :, 0] = px * c1 + py * s1
    pts[:, :, 1] = -px * s1 + py * c1
    tgt[:, :12] = pts.reshape(B, 12)
    return cam, sc, tgt


def _reason_mask(stop_reason: torch.Tensor, stop_reason_mask: torch.Tensor, name: str) -> torch.Tensor:
    if name not in STOP_REASON_NAMES:
        return torch.zeros_like(stop_reason.reshape(-1), dtype=torch.bool)
    active = _per_sample_vector(stop_reason_mask, int(stop_reason.reshape(-1).numel())) > 0.5
    return (stop_reason.reshape(-1) == int(STOP_REASON_NAMES.index(name))) & active


def _count_pair(metrics: dict[str, torch.Tensor], name: str, correct: torch.Tensor, active: torch.Tensor) -> None:
    active_f = active.reshape(-1).to(dtype=torch.float32)
    correct_f = correct.reshape(-1).to(dtype=torch.float32) * active_f
    metrics[f"{name}_num"] = correct_f.sum()
    metrics[f"{name}_den"] = active_f.sum()


def _miss_rate(metrics: dict[str, float], name: str, weight: float) -> float:
    count = float(metrics.get(f"{name}_count", 0.0))
    if count <= 0.0:
        return 0.0
    return float(weight) * max(0.0, 1.0 - float(metrics.get(name, 0.0)))


def _controller_proxy_losses(
    pred_traj: torch.Tensor,
    pred_speed: torch.Tensor,
    pred_stop_logit: torch.Tensor,
    scalar: torch.Tensor,
    control_target: torch.Tensor,
    control_mask: torch.Tensor,
    effective_weight: torch.Tensor,
    args,
) -> dict[str, torch.Tensor]:
    """Approximate the downstream TF++ controller and compare with expert controls."""

    device = pred_traj.device
    zero = torch.zeros((), dtype=torch.float32, device=device)
    metrics: dict[str, torch.Tensor] = {
        "controller_proxy": zero,
        "plan_steer_error": zero,
        "plan_throttle_error": zero,
        "plan_brake_error": zero,
    }
    for name in ("plan_brake_recall", "plan_go_recall", "plan_steer_close", "plan_throttle_close"):
        metrics[f"{name}_num"] = zero
        metrics[f"{name}_den"] = zero

    if control_target.numel() == 0 or control_target.ndim < 2 or control_target.shape[1] < 3:
        return metrics

    batch_size = int(pred_traj.shape[0])
    control_active_f = _per_sample_vector(control_mask, batch_size)
    control_active = control_active_f > 0.5
    if not bool(control_active.any()):
        return metrics

    expert_steer = control_target[:, 0].clamp(-1.0, 1.0)
    expert_throttle = control_target[:, 1].clamp(0.0, 1.0)
    expert_brake = control_target[:, 2].clamp(0.0, 1.0)

    waypoint_count = int(pred_traj.shape[1])
    aim_idx = min(max(waypoint_count - 1, 0), 3)
    aim = pred_traj[:, aim_idx, :2]
    pred_steer = torch.atan2(aim[:, 1], torch.clamp(aim[:, 0], min=1e-3)) / (0.5 * torch.pi)
    pred_steer = pred_steer.clamp(-1.0, 1.0)

    pred_speed_nonnegative = pred_speed.clamp_min(0.0)
    speed_from_head = pred_speed_nonnegative.amax(dim=1)
    if waypoint_count >= 2:
        one_idx = min(max(waypoint_count - 1, 1), 3)
        half_idx = max(0, one_idx // 2)
        plan_delta = pred_traj[:, one_idx, :2] - pred_traj[:, half_idx, :2]
        speed_from_plan = torch.linalg.norm(plan_delta, dim=1) * 2.0
    else:
        speed_from_plan = speed_from_head
    desired_speed = 0.5 * speed_from_head + 0.5 * speed_from_plan
    current_speed = scalar[:, 0].abs() if scalar.shape[1] else torch.zeros_like(desired_speed)

    pred_brake_bool = (
        (pred_stop_logit >= 0.0)
        | (speed_from_head <= float(args.stop_speed_ceiling_mps))
        | (desired_speed <= float(args.stop_speed_ceiling_mps))
    )
    pred_brake = pred_brake_bool.to(dtype=torch.float32)
    pred_throttle = ((desired_speed - current_speed) / max(float(args.speed_floor_mps), 1e-3)).clamp(0.0, 1.0)
    pred_throttle = pred_throttle * (1.0 - pred_brake)

    active_weight = effective_weight * control_active_f
    steer_raw = nn.functional.smooth_l1_loss(pred_steer, expert_steer, reduction="none")
    throttle_raw = nn.functional.smooth_l1_loss(pred_throttle, expert_throttle, reduction="none")
    brake_raw = nn.functional.smooth_l1_loss(pred_brake, expert_brake, reduction="none")
    metrics["plan_steer_error"] = _weighted_mean(steer_raw, active_weight)
    metrics["plan_throttle_error"] = _weighted_mean(throttle_raw, active_weight)
    metrics["plan_brake_error"] = _weighted_mean(brake_raw, active_weight)
    metrics["controller_proxy"] = (
        0.75 * metrics["plan_steer_error"]
        + 1.00 * metrics["plan_throttle_error"]
        + 2.00 * metrics["plan_brake_error"]
    )

    expert_brake_active = control_active & (expert_brake >= 0.5)
    expert_go_active = control_active & (expert_brake < 0.5) & (expert_throttle >= float(args.controller_go_throttle_threshold))
    steer_close = (pred_steer - expert_steer).abs() <= float(args.controller_steer_close_threshold)
    throttle_close = (pred_throttle - expert_throttle).abs() <= float(args.controller_throttle_close_threshold)
    _count_pair(metrics, "plan_brake_recall", pred_brake_bool, expert_brake_active)
    _count_pair(metrics, "plan_go_recall", ~pred_brake_bool & (pred_throttle >= float(args.controller_go_throttle_threshold)), expert_go_active)
    _count_pair(metrics, "plan_steer_close", steer_close, control_active)
    _count_pair(metrics, "plan_throttle_close", throttle_close, control_active)
    return metrics


class _BackboneTaskPatch:
    """Insert a trainable feature-then-fusion adapter inside one TF++ backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        adapter: FeatureThenFusionAdapter | ExtrinsicAwareFeatureThenFusionAdapter,
        stage_feature_shapes: Dict[str, tuple[int, int, int]],
        fused_feature_shape: tuple[int, int, int],
        stage_blend: float,
        fusion_blend: float,
        stage_adapter_layers: tuple[int, ...] | None = None,
        stage_adapter_modalities: str = "all",
        fusion_adapter_enabled: bool = True,
    ) -> None:
        self.backbone = backbone
        self.adapter = adapter
        self.stage_feature_shapes = stage_feature_shapes
        self.fused_feature_shape = fused_feature_shape
        self.stage_blend = float(stage_blend)
        self.fusion_blend = float(fusion_blend)
        self.stage_adapter_layers = None if stage_adapter_layers is None else tuple(int(v) for v in stage_adapter_layers)
        self.stage_adapter_modalities = str(stage_adapter_modalities)
        self.fusion_adapter_enabled = bool(fusion_adapter_enabled)
        self.original_fuse_features = backbone.fuse_features
        self.original_forward = backbone.forward
        self.enabled = True
        self.records: list[torch.Tensor] = []
        self.xmodal_records: list[torch.Tensor] = []
        _xm = _xmodal_config()
        self.xmodal_temp = float(_xm["temp"]) if float(_xm["weight"]) > 0.0 else None
        self.last_fused: torch.Tensor | None = None
        # v4 geometric-teacher distillation: when _capture_teacher is True the
        # original fuse path stores per-layer image features (from the reprojected
        # x=-1.5 view) into teacher_feats; the student pass then uses those as the
        # `_record` reference instead of its own un-adapted features.
        self.teacher_feats: dict[int, torch.Tensor] = {}
        self._capture_teacher = False
        self._install()

    def clear(self) -> None:
        self.records.clear()
        self.xmodal_records.clear()
        self.last_fused = None
        # NOTE: do NOT clear teacher_feats here. clear() is called between the
        # prior pass and the student pass (see forward), and the student pass needs
        # the teacher features captured earlier. teacher_feats is (re)initialized at
        # the top of forward() instead.
        self._capture_teacher = False

    def drift_loss(self, device: torch.device) -> torch.Tensor:
        if not self.records:
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack(self.records).mean()

    def xmodal_loss(self, device: torch.device) -> torch.Tensor:
        if not self.xmodal_records:
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack(self.xmodal_records).mean()

    def _record_xmodal(self, image_feat: torch.Tensor, lidar_feat: torch.Tensor, temp: float) -> None:
        # Per-frame InfoNCE between camera and lidar stage features: same-frame pairs
        # align (positives), cross-frame repel. Pulls the shifted camera to encode the
        # same scene content as the stable lidar. Global-pool -> [B, C] (C matches per layer).
        b = int(image_feat.shape[0])
        if b < 2:
            return
        zi = nn.functional.adaptive_avg_pool2d(image_feat.float(), 1).flatten(1)
        zl = nn.functional.adaptive_avg_pool2d(lidar_feat.float(), 1).flatten(1)
        zi = nn.functional.normalize(zi, dim=1)
        zl = nn.functional.normalize(zl, dim=1)
        logits = (zi @ zl.t()) / max(float(temp), 1e-4)
        labels = torch.arange(b, device=logits.device)
        loss = 0.5 * (
            nn.functional.cross_entropy(logits, labels)
            + nn.functional.cross_entropy(logits.t(), labels)
        )
        self.xmodal_records.append(loss)

    def restore(self) -> None:
        self.backbone.fuse_features = self.original_fuse_features
        self.backbone.forward = self.original_forward

    def _record(self, adapted: torch.Tensor, reference: torch.Tensor) -> None:
        self.records.append(nn.functional.smooth_l1_loss(adapted.float(), reference.detach().float()))

    def _install(self) -> None:
        backbone = self.backbone
        patch = self

        def adapted_fuse_features(image_features, lidar_features, layer_idx):
            if not patch.enabled:
                if patch._capture_teacher:
                    # teacher pass (adapter off): store the x=-1.5 view's per-layer
                    # image feature (same avgpool as the student path) as the
                    # distillation target for this fusion layer.
                    tidx = int(layer_idx)
                    patch.teacher_feats[tidx] = backbone.avgpool_img(image_features).detach()
                return patch.original_fuse_features(image_features, lidar_features, layer_idx)

            idx = int(layer_idx)
            if patch.stage_adapter_layers is not None and idx not in patch.stage_adapter_layers:
                return patch.original_fuse_features(image_features, lidar_features, layer_idx)
            if patch.stage_adapter_modalities == "none":
                return patch.original_fuse_features(image_features, lidar_features, layer_idx)

            image_embd_layer = backbone.avgpool_img(image_features)
            lidar_embd_layer = backbone.avgpool_lidar(lidar_features)
            lidar_embd_layer = backbone.lidar_channel_to_img[idx](lidar_embd_layer)

            image_name = f"layer_{idx}_image"
            lidar_name = f"layer_{idx}_lidar"
            expected_image = patch.stage_feature_shapes[image_name]
            expected_lidar = patch.stage_feature_shapes[lidar_name]
            got_image = tuple(int(v) for v in image_embd_layer.shape[1:])
            got_lidar = tuple(int(v) for v in lidar_embd_layer.shape[1:])
            if got_image != expected_image or got_lidar != expected_lidar:
                raise ValueError(
                    f"stage feature shape mismatch layer={idx}: "
                    f"image {got_image} != {expected_image}, lidar {got_lidar} != {expected_lidar}"
                )

            adapted_image, adapted_lidar = patch.adapter.adapt_layer(idx, image_embd_layer.float(), lidar_embd_layer.float())
            if patch.stage_adapter_modalities in {"all", "camera"}:
                # v4: distill toward the teacher (x=-1.5 reprojected view) feature when
                # available; otherwise fall back to the original self-reference (drift reg).
                cam_ref = patch.teacher_feats.get(idx, image_embd_layer)
                patch._record(adapted_image, cam_ref)
            else:
                adapted_image = image_embd_layer.float()
            if patch.stage_adapter_modalities in {"all", "lidar"}:
                patch._record(adapted_lidar, lidar_embd_layer)
            else:
                adapted_lidar = lidar_embd_layer.float()
            adapted_image = adapted_image.to(dtype=image_embd_layer.dtype)
            adapted_lidar = adapted_lidar.to(dtype=lidar_embd_layer.dtype)
            if patch.stage_blend < 1.0:
                adapted_image = image_embd_layer + patch.stage_blend * (adapted_image - image_embd_layer)
                adapted_lidar = lidar_embd_layer + patch.stage_blend * (adapted_lidar - lidar_embd_layer)

            if patch.xmodal_temp is not None:
                patch._record_xmodal(adapted_image, adapted_lidar, patch.xmodal_temp)

            image_features_layer, lidar_features_layer = backbone.transformers[idx](adapted_image, adapted_lidar)
            lidar_features_layer = backbone.img_channel_to_lidar[idx](lidar_features_layer)
            image_features_layer = nn.functional.interpolate(
                image_features_layer,
                size=(image_features.shape[2], image_features.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            if backbone.lidar_video:
                lidar_features_layer = nn.functional.interpolate(
                    lidar_features_layer,
                    size=(lidar_features.shape[2], lidar_features.shape[3], lidar_features.shape[4]),
                    mode="trilinear",
                    align_corners=False,
                )
            else:
                lidar_features_layer = nn.functional.interpolate(
                    lidar_features_layer,
                    size=(lidar_features.shape[2], lidar_features.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            return image_features + image_features_layer, lidar_features + lidar_features_layer

        def adapted_forward(*args, **kwargs):
            output = patch.original_forward(*args, **kwargs)
            if not patch.enabled:
                return output
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                return output
            fused = output[1]
            if not patch.fusion_adapter_enabled:
                patch.last_fused = fused
                return output
            got = tuple(int(v) for v in fused.shape[1:])
            if got != patch.fused_feature_shape:
                raise ValueError(f"fused feature shape mismatch: {got} != {patch.fused_feature_shape}")
            adapted = patch.adapter.adapt_fused(fused.float())
            patch._record(adapted, fused)
            adapted = adapted.to(dtype=fused.dtype)
            if patch.fusion_blend < 1.0:
                adapted = fused + patch.fusion_blend * (adapted - fused)
            patch.last_fused = adapted
            if isinstance(output, tuple):
                return (output[0], adapted, *output[2:])
            out = list(output)
            out[1] = adapted
            return out

        backbone.fuse_features = adapted_fuse_features
        backbone.forward = adapted_forward


class _AuxHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        hidden = int(hidden_dim) if int(hidden_dim) > 0 else max(64, int(input_dim) // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, int(output_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedOutputResidualHead(nn.Module):
    """Small bounded residual head for TF++ controller-facing outputs.

    The head does not replace the TransFuser++ policy. It predicts a gated,
    bounded delta for the checkpoint trajectory and target-speed logits, so the
    frozen TF++ prior remains the default behavior unless the residual branch
    learns a confident correction.
    """

    def __init__(
        self,
        fused_channels: int,
        checkpoint_dim: int,
        speed_classes: int,
        hidden_dim: int,
        checkpoint_scale: float,
        speed_logit_scale: float,
        gate_bias: float = -2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fused_channels = int(fused_channels)
        self.checkpoint_dim = int(checkpoint_dim)
        self.speed_classes = int(speed_classes)
        self.checkpoint_scale = float(checkpoint_scale)
        self.speed_logit_scale = float(speed_logit_scale)
        input_dim = self.fused_channels + self.checkpoint_dim + self.speed_classes
        hidden = int(hidden_dim) if int(hidden_dim) > 0 else max(128, input_dim // 2)
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.checkpoint_delta = nn.Linear(hidden, self.checkpoint_dim)
        self.speed_delta = nn.Linear(hidden, self.speed_classes)
        self.gate = nn.Linear(hidden, 2)
        nn.init.zeros_(self.checkpoint_delta.weight)
        nn.init.zeros_(self.checkpoint_delta.bias)
        nn.init.zeros_(self.speed_delta.weight)
        nn.init.zeros_(self.speed_delta.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias))

    @staticmethod
    def _pool_fused(fused: torch.Tensor) -> torch.Tensor:
        if fused.ndim == 4:
            return fused.mean(dim=(2, 3))
        if fused.ndim == 3:
            return fused.mean(dim=2)
        if fused.ndim == 2:
            return fused
        return fused.reshape(fused.shape[0], -1)

    @staticmethod
    def _pad_or_trim(x: torch.Tensor, width: int) -> torch.Tensor:
        width = int(width)
        if x.shape[1] == width:
            return x
        if x.shape[1] > width:
            return x[:, :width]
        pad = torch.zeros((x.shape[0], width - x.shape[1]), dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=1)

    def forward(
        self,
        fused: torch.Tensor,
        pred_checkpoint: torch.Tensor | None,
        pred_target_speed: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]]:
        batch_size = int(fused.shape[0])
        pooled = self._pad_or_trim(self._pool_fused(fused.float()), self.fused_channels)
        if pred_checkpoint is None:
            checkpoint_flat = torch.zeros((batch_size, self.checkpoint_dim), dtype=pooled.dtype, device=pooled.device)
            checkpoint_shape = None
        else:
            checkpoint_shape = tuple(int(v) for v in pred_checkpoint.shape)
            checkpoint_flat = self._pad_or_trim(pred_checkpoint.float().reshape(batch_size, -1), self.checkpoint_dim)
        if pred_target_speed is None:
            speed_logits = torch.zeros((batch_size, self.speed_classes), dtype=pooled.dtype, device=pooled.device)
        else:
            speed_logits = self._pad_or_trim(pred_target_speed.float(), self.speed_classes)

        hidden = self.trunk(torch.cat([pooled, checkpoint_flat, speed_logits], dim=1))
        gates = torch.sigmoid(self.gate(hidden))
        checkpoint_delta = torch.tanh(self.checkpoint_delta(hidden)) * float(self.checkpoint_scale) * gates[:, :1]
        speed_delta = torch.tanh(self.speed_delta(hidden)) * float(self.speed_logit_scale) * gates[:, 1:2]

        adapted_checkpoint = None
        if pred_checkpoint is not None and checkpoint_shape is not None:
            adapted_flat = checkpoint_flat + checkpoint_delta
            adapted_checkpoint = adapted_flat[:, : int(np.prod(checkpoint_shape[1:]))].reshape(checkpoint_shape)
            adapted_checkpoint = adapted_checkpoint.to(dtype=pred_checkpoint.dtype)
        adapted_speed = None
        if pred_target_speed is not None:
            adapted_speed = (speed_logits + speed_delta)[:, : pred_target_speed.shape[1]].to(dtype=pred_target_speed.dtype)

        stats = {
            "output_residual_checkpoint_gate": gates[:, 0].mean(),
            "output_residual_speed_gate": gates[:, 1].mean(),
            "output_residual_checkpoint_norm": checkpoint_delta.abs().mean(),
            "output_residual_speed_norm": speed_delta.abs().mean(),
        }
        return adapted_checkpoint, adapted_speed, stats


class _TaskAdapterForward(nn.Module):
    """Run frozen TF++ with an inserted task adapter in a DataParallel-safe wrapper."""

    def __init__(
        self,
        net: nn.Module,
        adapter: FeatureThenFusionAdapter | ExtrinsicAwareFeatureThenFusionAdapter,
        config,
        cameras: list[str],
        command_mode: str,
        tfpp_camera: str,
        stage_feature_shapes: Dict[str, tuple[int, int, int]],
        fused_feature_shape: tuple[int, int, int],
        stage_blend: float,
        fusion_blend: float,
        stage_adapter_layers: tuple[int, ...] | None = None,
        stage_adapter_modalities: str = "all",
        fusion_adapter_enabled: bool = True,
        aux_hidden_dim: int = 256,
        use_stop_state_aux: bool = False,
        use_stop_reason_aux: bool = False,
        use_control_aux: bool = False,
        use_output_residual: bool = False,
        output_residual_hidden_dim: int = 256,
        output_residual_checkpoint_scale: float = 0.75,
        output_residual_speed_logit_scale: float = 1.5,
        output_residual_gate_bias: float = -2.0,
        output_residual_dropout: float = 0.0,
        camera_crop_shift_x_px: float = 0.0,
        camera_crop_shift_y_px: float = 0.0,
        camera_crop_scale: float = 1.0,
        camera_ground_plane_warp: bool = False,
        camera_ground_plane_source_pose: tuple[float, ...] = (1.25, 0.0, 1.95, 0.0, 0.0, 0.0),
        camera_ground_plane_target_pose: tuple[float, ...] = (-1.5, 0.0, 2.0, 0.0, 0.0, 0.0),
        camera_ground_plane_z_m: float = 0.0,
        lidar_shift_x_m: float = 0.0,
        lidar_shift_y_m: float = 0.0,
        lidar_pixels_per_meter: float = 4.0,
    ) -> None:
        super().__init__()
        self.net = net
        self.adapter = adapter
        self.config = config
        self.cameras = list(cameras)
        self.command_mode = str(command_mode)
        self.tfpp_camera = str(tfpp_camera)
        self.stage_feature_shapes = stage_feature_shapes
        self.fused_feature_shape = fused_feature_shape
        self.stage_blend = float(stage_blend)
        self.fusion_blend = float(fusion_blend)
        self.stage_adapter_layers = stage_adapter_layers
        self.stage_adapter_modalities = str(stage_adapter_modalities)
        self.fusion_adapter_enabled = bool(fusion_adapter_enabled)
        self.camera_crop_shift_x_px = float(camera_crop_shift_x_px)
        self.camera_crop_shift_y_px = float(camera_crop_shift_y_px)
        self.camera_crop_scale = float(camera_crop_scale)
        self.camera_ground_plane_warp = bool(camera_ground_plane_warp)
        self.camera_ground_plane_source_pose = tuple(float(v) for v in camera_ground_plane_source_pose)
        self.camera_ground_plane_target_pose = tuple(float(v) for v in camera_ground_plane_target_pose)
        self.camera_ground_plane_z_m = float(camera_ground_plane_z_m)
        self.lidar_shift_x_m = float(lidar_shift_x_m)
        self.lidar_shift_y_m = float(lidar_shift_y_m)
        self.lidar_pixels_per_meter = float(lidar_pixels_per_meter)
        self._patch: _BackboneTaskPatch | None = None
        fused_channels = int(fused_feature_shape[0])
        self.stop_state_head = _AuxHead(fused_channels, 4, int(aux_hidden_dim)) if use_stop_state_aux else None
        self.stop_reason_head = _AuxHead(fused_channels, 8, int(aux_hidden_dim)) if use_stop_reason_aux else None
        self.control_head = _AuxHead(fused_channels, 3, int(aux_hidden_dim)) if use_control_aux else None
        checkpoint_len = int(getattr(config, "predict_checkpoint_len", 10) or 10)
        speed_classes = len(getattr(config, "target_speeds", [])) or 1
        self.output_residual_head = (
            GatedOutputResidualHead(
                fused_channels=fused_channels,
                checkpoint_dim=checkpoint_len * 2,
                speed_classes=speed_classes,
                hidden_dim=int(output_residual_hidden_dim),
                checkpoint_scale=float(output_residual_checkpoint_scale),
                speed_logit_scale=float(output_residual_speed_logit_scale),
                gate_bias=float(output_residual_gate_bias),
                dropout=float(output_residual_dropout),
            )
            if use_output_residual
            else None
        )

    def set_runtime_mode(self, train: bool) -> None:
        self.net.eval()
        self.adapter.train(train)
        for head in (self.stop_state_head, self.stop_reason_head, self.control_head, self.output_residual_head):
            if head is not None:
                head.train(train)
        set_lora_train_mode(self.net, train)
        if train:
            for module in self.net.modules():
                if isinstance(module, nn.modules.rnn.RNNBase):
                    module.train(True)
            # v5: let BatchNorm track target-domain stats during training (AdaBN-in-the-loop)
            if os.environ.get("T2D_BN_TRAIN_MODE", "0") == "1":
                for module in self.net.modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                        module.train(True)

    def _ensure_patch(self) -> _BackboneTaskPatch:
        if self._patch is None:
            self._patch = _BackboneTaskPatch(
                self.net.backbone,
                self.adapter,
                self.stage_feature_shapes,
                self.fused_feature_shape,
                stage_blend=self.stage_blend,
                fusion_blend=self.fusion_blend,
                stage_adapter_layers=self.stage_adapter_layers,
                stage_adapter_modalities=self.stage_adapter_modalities,
                fusion_adapter_enabled=self.fusion_adapter_enabled,
            )
        return self._patch

    def restore(self) -> None:
        if self._patch is not None:
            self._patch.restore()
            self._patch = None

    def aux_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        state: dict[str, dict[str, torch.Tensor]] = {}
        if self.stop_state_head is not None:
            state["stop_state_head"] = self.stop_state_head.state_dict()
        if self.stop_reason_head is not None:
            state["stop_reason_head"] = self.stop_reason_head.state_dict()
        if self.control_head is not None:
            state["control_head"] = self.control_head.state_dict()
        if self.output_residual_head is not None:
            state["output_residual_head"] = self.output_residual_head.state_dict()
        return state

    @staticmethod
    def _pool_aux_feature(fused: torch.Tensor) -> torch.Tensor:
        if fused.ndim == 4:
            return fused.mean(dim=(2, 3))
        if fused.ndim == 3:
            return fused.mean(dim=2)
        if fused.ndim == 2:
            return fused
        return fused.reshape(fused.shape[0], -1)

    def forward(
        self,
        scalar: torch.Tensor,
        camera: torch.Tensor,
        lidar: torch.Tensor,
        target_dim: int,
        speed_dim: int,
        camera_teacher: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        patch = self._ensure_patch()
        inputs = prepare_transfuserpp_inputs(
            scalar=scalar,
            camera=camera,
            lidar=lidar,
            cameras=self.cameras,
            config=self.config,
            command_mode=self.command_mode,
            tfpp_camera=self.tfpp_camera,
            camera_crop_shift_x_px=self.camera_crop_shift_x_px,
            camera_crop_shift_y_px=self.camera_crop_shift_y_px,
            camera_crop_scale=self.camera_crop_scale,
            camera_ground_plane_warp=self.camera_ground_plane_warp,
            camera_ground_plane_source_pose=self.camera_ground_plane_source_pose,
            camera_ground_plane_target_pose=self.camera_ground_plane_target_pose,
            camera_ground_plane_z_m=self.camera_ground_plane_z_m,
            lidar_shift_x_m=self.lidar_shift_x_m,
            lidar_shift_y_m=self.lidar_shift_y_m,
            lidar_pixels_per_meter=self.lidar_pixels_per_meter,
        )

        # v4: capture geometric-teacher features from the reprojected x=-1.5 view.
        # Only the camera(rgb) is swapped; lidar/scalar identical -> the fusion
        # layers' image branch encodes the pretrained-rig viewpoint, which becomes
        # the _record distillation target for the student pass below.
        patch.teacher_feats = {}
        if camera_teacher is not None:
            teacher_inputs = prepare_transfuserpp_inputs(
                scalar=scalar,
                camera=camera_teacher,
                lidar=lidar,
                cameras=[self.tfpp_camera],
                config=self.config,
                command_mode=self.command_mode,
                tfpp_camera=self.tfpp_camera,
                camera_crop_shift_x_px=self.camera_crop_shift_x_px,
                camera_crop_shift_y_px=self.camera_crop_shift_y_px,
                camera_crop_scale=self.camera_crop_scale,
                camera_ground_plane_warp=self.camera_ground_plane_warp,
                camera_ground_plane_source_pose=self.camera_ground_plane_source_pose,
                camera_ground_plane_target_pose=self.camera_ground_plane_target_pose,
                camera_ground_plane_z_m=self.camera_ground_plane_z_m,
                lidar_shift_x_m=self.lidar_shift_x_m,
                lidar_shift_y_m=self.lidar_shift_y_m,
                lidar_pixels_per_meter=self.lidar_pixels_per_meter,
            )
            patch.enabled = False
            patch._capture_teacher = True
            try:
                with torch.no_grad():
                    self.net(**teacher_inputs)
            finally:
                patch._capture_teacher = False
                patch.enabled = True

        patch.enabled = False
        try:
            with torch.no_grad():
                prior_outputs = self.net(**inputs)
                prior_target = base_target_from_checkpoint(
                    pred_checkpoint=prior_outputs[2],
                    pred_target_speed=prior_outputs[1],
                    scalar=scalar,
                    config=self.config,
                    target_dim=int(target_dim),
                    speed_dim=int(speed_dim),
                ).detach()
        finally:
            patch.enabled = True
        patch.clear()
        outputs = self.net(**inputs)
        aux: dict[str, torch.Tensor] = {}
        # v12: expose the adapted (patch-enabled) perspective depth prediction so the
        # training loop can supervise it against lidar-projected depth. outputs[5] is
        # pred_depth = sigmoid(depth_decoder(image_feature_grid)) in [0, 1], shape [B,Hd,Wd].
        if len(outputs) > 5 and outputs[5] is not None:
            aux["pred_depth"] = outputs[5]
        pred_target_speed = outputs[1]
        pred_checkpoint = outputs[2]
        if self.output_residual_head is not None and patch.last_fused is not None:
            pred_checkpoint, pred_target_speed, residual_stats = self.output_residual_head(
                patch.last_fused.float(),
                pred_checkpoint,
                pred_target_speed,
            )
            aux.update(residual_stats)
        pred_target = base_target_from_checkpoint(
            pred_checkpoint=pred_checkpoint,
            pred_target_speed=pred_target_speed,
            scalar=scalar,
            config=self.config,
            target_dim=int(target_dim),
            speed_dim=int(speed_dim),
        )
        if patch.last_fused is not None and (
            self.stop_state_head is not None or self.stop_reason_head is not None or self.control_head is not None
        ):
            aux_feature = self._pool_aux_feature(patch.last_fused.float())
            if self.stop_state_head is not None:
                aux["stop_state"] = self.stop_state_head(aux_feature)
            if self.stop_reason_head is not None:
                aux["stop_reason"] = self.stop_reason_head(aux_feature)
            if self.control_head is not None:
                aux["control"] = self.control_head(aux_feature)
        return pred_target, patch.drift_loss(scalar.device).reshape(1), prior_target, aux


def _infer_feature_shapes(net: nn.Module, config, loader: DataLoader, cameras: list[str], args, device: torch.device):
    captured, restore = _capture_backbone_feature_fusion_features(net)
    try:
        batch = next(iter(loader))
        scalar = batch["scalar"].to(device, non_blocking=True)
        camera = batch["camera"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        inputs = prepare_transfuserpp_inputs(
            scalar=scalar,
            camera=camera,
            lidar=lidar,
            cameras=cameras,
            config=config,
            command_mode=args.command_mode,
            tfpp_camera=args.tfpp_camera,
            camera_crop_shift_x_px=float(args.camera_crop_shift_x_px),
            camera_crop_shift_y_px=float(args.camera_crop_shift_y_px),
            camera_crop_scale=float(args.camera_crop_scale),
            camera_ground_plane_warp=bool(args.camera_ground_plane_warp),
            camera_ground_plane_source_pose=tuple(float(v) for v in args.camera_ground_plane_source_pose),
            camera_ground_plane_target_pose=tuple(float(v) for v in args.camera_ground_plane_target_pose),
            camera_ground_plane_z_m=float(args.camera_ground_plane_z_m),
            lidar_shift_x_m=float(args.lidar_canonical_shift_x_m),
            lidar_shift_y_m=float(args.lidar_canonical_shift_y_m),
            lidar_pixels_per_meter=float(args.lidar_pixels_per_meter),
        )
        captured.clear()
        with torch.no_grad():
            _ = net(**inputs)
        missing = [name for name in ALL_FEATURE_NAMES if name not in captured]
        if missing:
            raise RuntimeError(f"Backbone capture missing features: {missing}")
        stage_shapes = {
            name: tuple(int(v) for v in captured[name].shape[1:])
            for name in ALL_FEATURE_NAMES
            if name != FUSED_FEATURE_NAME
        }
        fused_shape = tuple(int(v) for v in captured[FUSED_FEATURE_NAME].shape[1:])
        return stage_shapes, fused_shape
    finally:
        restore()


def _losses(
    pred: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    drift_loss: torch.Tensor,
    prior: torch.Tensor,
    aux: dict[str, torch.Tensor],
    device: torch.device,
    args,
):
    target = batch["target"].to(device, non_blocking=True)
    scalar = batch["scalar"].to(device, non_blocking=True)
    stop_state = batch["stop_state"].to(device, non_blocking=True)
    stop_reason = batch["stop_reason"].to(device, non_blocking=True)
    stop_reason_mask = batch["stop_reason_mask"].to(device, non_blocking=True)
    control_target = batch["control_target"].to(device, non_blocking=True)
    control_mask = batch["control_mask"].to(device, non_blocking=True)
    weight = batch["sample_weight"].to(device, non_blocking=True)

    speed_dim = int(args.speed_dim)
    traj_dim = target.shape[1] - speed_dim - 1
    pred_traj = pred[:, :traj_dim].reshape(pred.shape[0], -1, 3)
    target_traj = target[:, :traj_dim].reshape(target.shape[0], -1, 3)
    pred_speed = pred[:, traj_dim : traj_dim + speed_dim]
    target_speed = target[:, traj_dim : traj_dim + speed_dim]
    prior_traj = prior[:, :traj_dim].reshape(prior.shape[0], -1, 3)
    prior_speed_target = prior[:, traj_dim : traj_dim + speed_dim]
    stop_target = target[:, -1:].clamp(0.0, 1.0)

    moving = _moving_mask(scalar, target, speed_dim, float(args.moving_speed_threshold))
    launch = _launch_mask(scalar, target_speed, float(args.launch_current_speed_threshold), float(args.launch_target_speed_threshold))
    release = _release_mask(stop_state, target_speed, float(args.release_target_speed_threshold))
    hazard = _hazard_reason_mask(stop_reason, args.hazard_stop_reasons)

    effective_weight = _per_sample_vector(weight, int(target.shape[0]))
    effective_weight = effective_weight * torch.where(
        moving,
        torch.full_like(effective_weight, float(args.moving_sample_weight)),
        torch.full_like(effective_weight, float(args.stopped_sample_weight)),
    )
    effective_weight = effective_weight * torch.where(
        hazard,
        torch.full_like(effective_weight, float(args.hazard_sample_weight)),
        torch.ones_like(effective_weight),
    )
    effective_weight = effective_weight * torch.where(
        launch,
        torch.full_like(effective_weight, float(args.launch_sample_weight)),
        torch.ones_like(effective_weight),
    )
    effective_weight = effective_weight * torch.where(
        release,
        torch.full_like(effective_weight, float(args.release_sample_weight)),
        torch.ones_like(effective_weight),
    )

    xy = _weighted_mean(
        torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., :2], target_traj[..., :2], reduction="none"), dim=(1, 2)),
        effective_weight,
    )
    yaw = _weighted_mean(
        torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., 2], target_traj[..., 2], reduction="none"), dim=1),
        effective_weight,
    )
    speed = _weighted_mean(
        torch.mean(nn.functional.smooth_l1_loss(pred_speed, target_speed, reduction="none"), dim=1),
        effective_weight,
    )
    if pred_traj.shape[1] >= 3:
        pred_xy_accel = pred_traj[:, 2:, :2] - 2.0 * pred_traj[:, 1:-1, :2] + pred_traj[:, :-2, :2]
        target_xy_accel = target_traj[:, 2:, :2] - 2.0 * target_traj[:, 1:-1, :2] + target_traj[:, :-2, :2]
        traj_smooth = _weighted_mean(
            torch.mean(nn.functional.smooth_l1_loss(pred_xy_accel, target_xy_accel, reduction="none"), dim=(1, 2)),
            effective_weight,
        )
    else:
        traj_smooth = torch.zeros((), dtype=torch.float32, device=device)
    if pred_speed.shape[1] >= 2:
        pred_speed_delta = pred_speed[:, 1:] - pred_speed[:, :-1]
        target_speed_delta = target_speed[:, 1:] - target_speed[:, :-1]
        speed_smooth = _weighted_mean(
            torch.mean(nn.functional.smooth_l1_loss(pred_speed_delta, target_speed_delta, reduction="none"), dim=1),
            effective_weight,
        )
    else:
        speed_smooth = torch.zeros((), dtype=torch.float32, device=device)
    target_go = _target_speed_mask(target_speed, float(args.speed_floor_target_threshold))
    target_stop = ~_target_speed_mask(target_speed, float(args.stop_speed_target_threshold))
    speed_floor = _weighted_mean(
        torch.relu(float(args.speed_floor_mps) - pred_speed).mean(dim=1) * target_go.to(dtype=pred_speed.dtype),
        effective_weight,
    )
    launch_floor = _weighted_mean(
        torch.relu(float(args.launch_speed_floor_mps) - pred_speed).mean(dim=1) * launch.to(dtype=pred_speed.dtype),
        effective_weight,
    )
    release_floor = _weighted_mean(
        torch.relu(float(args.release_speed_floor_mps) - pred_speed).mean(dim=1) * release.to(dtype=pred_speed.dtype),
        effective_weight,
    )
    stop_ceiling = _weighted_mean(
        torch.relu(pred_speed - float(args.stop_speed_ceiling_mps)).mean(dim=1) * target_stop.to(dtype=pred_speed.dtype),
        effective_weight,
    )
    pred_speed_max = pred_speed.clamp_min(0.0).amax(dim=1)
    pred_stop_logit = pred[:, -1]
    stop_active = target_stop.reshape(-1).bool()
    go_active = target_go.reshape(-1).bool()
    launch_active = launch.reshape(-1).bool()
    release_active = release.reshape(-1).bool()
    hazard_active = hazard.reshape(-1).bool()
    pred_stop_by_speed = pred_speed_max <= float(args.stop_speed_ceiling_mps)
    pred_go_by_speed = pred_speed_max >= float(args.speed_floor_mps)
    pred_launch_by_speed = pred_speed_max >= float(args.launch_speed_floor_mps)
    pred_release_by_speed = pred_speed_max >= float(args.release_speed_floor_mps)

    stop_target_label = stop_target.reshape(-1) >= 0.5
    pred_stop_by_logit = pred_stop_logit >= 0.0
    stop_logit_acc_raw = pred_stop_by_logit == stop_target_label

    target_final_xy = target_traj[:, -1, :2]
    pred_final_xy = pred_traj[:, -1, :2]
    target_progress = torch.linalg.norm(target_final_xy, dim=1)
    pred_displacement = torch.linalg.norm(pred_final_xy, dim=1)
    unit_target = target_final_xy / torch.clamp(target_progress[:, None], min=1e-6)
    pred_progress = torch.sum(pred_final_xy * unit_target, dim=1)
    pred_lateral = pred_final_xy - pred_progress[:, None] * unit_target
    lateral_error_raw = torch.linalg.norm(pred_lateral, dim=1)
    lateral_error = _weighted_mean(lateral_error_raw, effective_weight)
    progress_error_raw = nn.functional.smooth_l1_loss(pred_progress, target_progress, reduction="none")
    progress_error = _weighted_mean(progress_error_raw, effective_weight)
    go_progress_ok = pred_progress >= (target_progress * float(args.go_progress_ratio))
    stop_progress_ok = pred_displacement <= float(args.stop_progress_ceiling_m)
    hazard_stop_ceiling = _weighted_mean(
        torch.relu(pred_speed - float(args.stop_speed_ceiling_mps)).mean(dim=1) * hazard_active.to(dtype=pred_speed.dtype),
        effective_weight,
    )
    hazard_progress = _weighted_mean(
        torch.relu(pred_displacement - float(args.stop_progress_ceiling_m)) * hazard_active.to(dtype=pred_speed.dtype),
        effective_weight,
    )
    behavior_proxy = (
        stop_ceiling
        + speed_floor
        + launch_floor
        + release_floor
        + hazard_stop_ceiling
        + 0.25 * progress_error
        + 0.25 * hazard_progress
    )
    stop = _weighted_mean(nn.functional.binary_cross_entropy_with_logits(pred[:, -1:], stop_target, reduction="none"), effective_weight)
    drift = drift_loss.reshape(-1).mean().to(device=device, dtype=torch.float32)
    prior_xy = _weighted_mean(
        torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., :2], prior_traj[..., :2], reduction="none"), dim=(1, 2)),
        effective_weight,
    )
    prior_speed = _weighted_mean(
        torch.mean(nn.functional.smooth_l1_loss(pred_speed, prior_speed_target, reduction="none"), dim=1),
        effective_weight,
    )
    controller_metrics = _controller_proxy_losses(
        pred_traj=pred_traj,
        pred_speed=pred_speed,
        pred_stop_logit=pred_stop_logit,
        scalar=scalar,
        control_target=control_target,
        control_mask=control_mask,
        effective_weight=effective_weight,
        args=args,
    )
    zero = torch.zeros((), dtype=torch.float32, device=device)
    control = zero
    if "control" in aux and control_target.numel() > 0:
        control_raw = torch.mean(nn.functional.smooth_l1_loss(aux["control"], control_target, reduction="none"), dim=1)
        control_active = _per_sample_vector(control_mask, int(control_raw.numel())) * effective_weight
        control = torch.sum(control_raw * control_active) / torch.clamp(torch.sum(control_active), min=1e-6)
    residual_checkpoint_gate = aux.get("output_residual_checkpoint_gate", zero)
    residual_speed_gate = aux.get("output_residual_speed_gate", zero)
    residual_checkpoint_norm = aux.get("output_residual_checkpoint_norm", zero)
    residual_speed_norm = aux.get("output_residual_speed_norm", zero)

    stop_state_aux = zero
    if "stop_state" in aux:
        stop_state_aux = _weighted_mean(nn.functional.cross_entropy(aux["stop_state"], stop_state.reshape(-1), reduction="none"), effective_weight)

    stop_reason_aux = zero
    if "stop_reason" in aux:
        reason_raw = nn.functional.cross_entropy(aux["stop_reason"], stop_reason.reshape(-1), reduction="none")
        reason_active = _per_sample_vector(stop_reason_mask, int(reason_raw.numel())) * effective_weight
        stop_reason_aux = torch.sum(reason_raw * reason_active) / torch.clamp(torch.sum(reason_active), min=1e-6)

    total = (
        float(args.xy_loss_weight) * xy
        + float(args.yaw_loss_weight) * yaw
        + float(args.speed_loss_weight) * speed
        + float(args.traj_smooth_loss_weight) * traj_smooth
        + float(args.speed_smooth_loss_weight) * speed_smooth
        + float(args.speed_floor_loss_weight) * speed_floor
        + float(args.launch_speed_floor_loss_weight) * launch_floor
        + float(args.release_speed_floor_loss_weight) * release_floor
        + float(args.stop_speed_ceiling_loss_weight) * stop_ceiling
        + float(args.stop_loss_weight) * stop
        + float(args.feature_drift_loss_weight) * drift
        + float(args.output_prior_xy_loss_weight) * prior_xy
        + float(args.output_prior_speed_loss_weight) * prior_speed
        + float(args.control_loss_weight) * control
        + float(args.stop_state_aux_loss_weight) * stop_state_aux
        + float(args.stop_reason_aux_loss_weight) * stop_reason_aux
        + float(args.pdm_behavior_loss_weight) * behavior_proxy
        + float(args.pdm_lateral_loss_weight) * lateral_error
        + float(args.pdm_progress_loss_weight) * progress_error
        + float(args.pdm_hazard_progress_loss_weight) * hazard_progress
        + float(args.pdm_controller_loss_weight) * controller_metrics["controller_proxy"]
        + float(args.pdm_plan_steer_loss_weight) * controller_metrics["plan_steer_error"]
        + float(args.pdm_plan_throttle_loss_weight) * controller_metrics["plan_throttle_error"]
        + float(args.pdm_plan_brake_loss_weight) * controller_metrics["plan_brake_error"]
    )
    metrics = {
        "loss": total,
        "xy": xy,
        "yaw": yaw,
        "speed": speed,
        "traj_smooth": traj_smooth,
        "speed_smooth": speed_smooth,
        "speed_floor": speed_floor,
        "launch_floor": launch_floor,
        "release_floor": release_floor,
        "stop_ceiling": stop_ceiling,
        "hazard_stop_ceiling": hazard_stop_ceiling,
        "lateral_error": lateral_error,
        "progress_error": progress_error,
        "hazard_progress": hazard_progress,
        "behavior_proxy": behavior_proxy,
        "stop": stop,
        "drift": drift,
        "prior_xy": prior_xy,
        "prior_speed": prior_speed,
        "controller_proxy": controller_metrics["controller_proxy"],
        "plan_steer_error": controller_metrics["plan_steer_error"],
        "plan_throttle_error": controller_metrics["plan_throttle_error"],
        "plan_brake_error": controller_metrics["plan_brake_error"],
        "control": control,
        "output_residual_checkpoint_gate": residual_checkpoint_gate,
        "output_residual_speed_gate": residual_speed_gate,
        "output_residual_checkpoint_norm": residual_checkpoint_norm,
        "output_residual_speed_norm": residual_speed_norm,
        "stop_state_aux": stop_state_aux,
        "stop_reason_aux": stop_reason_aux,
        "moving_ratio": moving.float().mean(),
        "hazard_ratio": hazard.float().mean(),
        "launch_ratio": launch.float().mean(),
        "release_ratio": release.float().mean(),
    }
    _count_pair(metrics, "stop_hold_recall", pred_stop_by_speed, stop_active)
    _count_pair(metrics, "go_recall", pred_go_by_speed, go_active)
    _count_pair(metrics, "hazard_hold_recall", pred_stop_by_speed, hazard_active)
    _count_pair(metrics, "launch_go_recall", pred_launch_by_speed, launch_active)
    _count_pair(metrics, "release_go_recall", pred_release_by_speed, release_active)
    _count_pair(metrics, "stop_logit_recall", pred_stop_by_logit, stop_target_label)
    _count_pair(metrics, "stop_logit_go_recall", ~pred_stop_by_logit, ~stop_target_label)
    _count_pair(metrics, "stop_logit_acc", stop_logit_acc_raw, torch.ones_like(stop_target_label, dtype=torch.bool))
    _count_pair(metrics, "go_progress_recall", go_progress_ok, go_active)
    _count_pair(metrics, "stop_progress_hold", stop_progress_ok, stop_active)
    _count_pair(metrics, "hazard_progress_hold", stop_progress_ok, hazard_active)
    for reason_name in ("traffic_light", "stop_sign", "front_vehicle", "junction_yield"):
        reason_active = _reason_mask(stop_reason, stop_reason_mask, reason_name).to(device=device)
        metric_name = reason_name.replace("_", "")
        _count_pair(metrics, f"{metric_name}_hold_recall", pred_stop_by_speed, reason_active)
    for name in ("plan_brake_recall", "plan_go_recall", "plan_steer_close", "plan_throttle_close"):
        metrics[f"{name}_num"] = controller_metrics[f"{name}_num"]
        metrics[f"{name}_den"] = controller_metrics[f"{name}_den"]
    return metrics


def _set_frozen_tfpp_runtime_mode(net: nn.Module, train: bool) -> None:
    """Keep TF++ deterministic, but allow cuDNN RNN backward during training."""

    net.eval()
    set_lora_train_mode(net, train)
    if not train:
        return
    for module in net.modules():
        if isinstance(module, nn.modules.rnn.RNNBase):
            module.train(True)


def _raw_task_model(model: nn.Module) -> _TaskAdapterForward:
    return model.module if isinstance(model, nn.DataParallel) else model


def _load_task_checkpoint_extras(model: _TaskAdapterForward, checkpoint_path: str, strict: bool = False) -> dict[str, dict]:
    if not checkpoint_path:
        return {}
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    info: dict[str, dict] = {}
    lora_state = checkpoint.get("peft_lora_state") or {}
    if lora_state:
        info["peft_lora_state"] = load_lora_state_dict(model.net, lora_state, strict=strict)

    aux_state = checkpoint.get("aux_state") or {}
    aux_info: dict[str, dict[str, int]] = {}
    for key in ("stop_state_head", "stop_reason_head", "control_head", "output_residual_head"):
        module = getattr(model, key, None)
        if module is None or key not in aux_state:
            continue
        missing, unexpected = module.load_state_dict(aux_state[key], strict=strict)
        aux_info[key] = {"missing": len(missing), "unexpected": len(unexpected)}
    if aux_info:
        info["aux_state"] = aux_info
    return info


def _capture_trainable_anchor(model: nn.Module, weight: float) -> dict[str, torch.Tensor]:
    if float(weight) <= 0.0:
        return {}
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _trainable_anchor_loss(model: nn.Module, anchor: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    if not anchor:
        return torch.zeros((), dtype=torch.float32, device=device)
    losses = []
    for name, param in model.named_parameters():
        ref = anchor.get(name)
        if ref is None or not param.requires_grad:
            continue
        losses.append(nn.functional.smooth_l1_loss(param.float(), ref.to(device=device, dtype=param.dtype).float()))
    if not losses:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(losses).mean()


def _run_epoch(model, loader, optimizer, device, args, train: bool, epoch: int, force_perturb: bool = False):
    raw_model = _raw_task_model(model)
    raw_model.set_runtime_mode(train)
    metric_names = (
        "loss",
        "xy",
        "yaw",
        "speed",
        "traj_smooth",
        "speed_smooth",
        "speed_floor",
        "launch_floor",
        "release_floor",
        "stop_ceiling",
        "hazard_stop_ceiling",
        "lateral_error",
        "progress_error",
        "hazard_progress",
        "behavior_proxy",
        "stop",
        "drift",
        "prior_xy",
        "prior_speed",
        "controller_proxy",
        "plan_steer_error",
        "plan_throttle_error",
        "plan_brake_error",
        "control",
        "output_residual_checkpoint_gate",
        "output_residual_speed_gate",
        "output_residual_checkpoint_norm",
        "output_residual_speed_norm",
        "stop_state_aux",
        "stop_reason_aux",
        "moving_ratio",
        "hazard_ratio",
        "launch_ratio",
        "release_ratio",
        "anchor",
        "xmodal",
        "depth",
    )
    rate_metric_names = (
        "stop_hold_recall",
        "go_recall",
        "hazard_hold_recall",
        "launch_go_recall",
        "release_go_recall",
        "stop_logit_recall",
        "stop_logit_go_recall",
        "stop_logit_acc",
        "go_progress_recall",
        "stop_progress_hold",
        "hazard_progress_hold",
        "trafficlight_hold_recall",
        "stopsign_hold_recall",
        "frontvehicle_hold_recall",
        "junctionyield_hold_recall",
        "plan_brake_recall",
        "plan_go_recall",
        "plan_steer_close",
        "plan_throttle_close",
    )
    totals = {name: 0.0 for name in metric_names}
    for name in rate_metric_names:
        totals[f"{name}_num"] = 0.0
        totals[f"{name}_den"] = 0.0
    totals["samples"] = 0
    start = time.time()
    perturb_cfg = _recovery_perturb_config()
    if force_perturb:
        # deterministic off-pose validation metric: same perturbation every epoch -> comparable,
        # and (unlike expert-pose metrics) it reflects off-distribution robustness that predicts
        # closed-loop. prob=1.0 so every val sample is evaluated off-pose.
        perturb_cfg = dict(perturb_cfg)
        perturb_cfg["prob"] = 1.0
        torch.manual_seed(20260713)
    for step, batch in enumerate(loader, start=1):
        scalar = batch["scalar"].to(device, non_blocking=True)
        camera = batch["camera"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        camera_teacher = batch.get("camera_teacher")
        if camera_teacher is not None:
            camera_teacher = camera_teacher.to(device, non_blocking=True)
        loss_batch = batch
        if (train or force_perturb) and float(perturb_cfg["prob"]) > 0.0:
            camera, scalar, _pt = _apply_recovery_perturbation(
                camera, scalar, batch["target"].to(device, non_blocking=True), perturb_cfg
            )
            loss_batch = dict(batch)
            loss_batch["target"] = _pt
        with torch.set_grad_enabled(train):
            pred_target, drift_loss, prior_target, aux = model(
                scalar,
                camera,
                lidar,
                target_dim=loss_batch["target"].shape[1],
                speed_dim=int(args.speed_dim),
                camera_teacher=camera_teacher,
            )
            losses = _losses(pred_target, loss_batch, drift_loss, prior_target, aux, device, args)
            anchor_loss = _trainable_anchor_loss(model, getattr(args, "_trainable_anchor_state", {}), device)
            losses["anchor"] = anchor_loss
            if train and float(args.init_param_anchor_loss_weight) > 0.0:
                losses["loss"] = losses["loss"] + float(args.init_param_anchor_loss_weight) * anchor_loss
            # v11: cross-modal (camera<->lidar) alignment -- re-anchor shifted camera to stable lidar
            _xm_w = float(_xmodal_config()["weight"])
            _raw = _raw_task_model(model)
            _xm_loss = _raw._patch.xmodal_loss(device) if (_xm_w > 0.0 and getattr(_raw, "_patch", None) is not None) else torch.zeros((), device=device)
            losses["xmodal"] = _xm_loss.detach()
            if train and _xm_w > 0.0:
                losses["loss"] = losses["loss"] + _xm_w * _xm_loss
            # v12: HARD lidar-distance injection. Supervise the camera perspective-depth
            # head against lidar-projected sparse depth GT (0 == invalid). This re-teaches
            # the camera correct metric distances from the shifted (2.75 m forward) viewpoint
            # -- especially oncoming-vehicle distance, the left-turn failure mode. The frozen
            # depth head acts as a fixed distance readout, so the gradient shapes the (unfrozen)
            # backbone + fusion to encode geometrically correct distance.
            _depth_w = float(os.environ.get("T2D_DEPTH_WEIGHT", "0") or 0.0)
            _depth_loss = torch.zeros((), device=device)
            _pred_depth = aux.get("pred_depth") if isinstance(aux, dict) else None
            if _depth_w > 0.0 and _pred_depth is not None and "depth_gt" in loss_batch:
                _dg = loss_batch["depth_gt"].to(device, non_blocking=True).float()  # [B,Hg,Wg]
                _pd = _pred_depth.float()
                if _pd.dim() == 4:
                    _pd = _pd.squeeze(1)
                _pd = nn.functional.interpolate(
                    _pd.unsqueeze(1), size=_dg.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)
                _mask = _dg > 0
                if bool(_mask.any()):
                    _depth_loss = nn.functional.l1_loss(_pd[_mask], _dg[_mask])
            losses["depth"] = _depth_loss.detach()
            if train and _depth_w > 0.0:
                losses["loss"] = losses["loss"] + _depth_w * _depth_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if float(args.grad_clip) > 0.0:
                    nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        float(args.grad_clip),
                    )
                optimizer.step()
        batch_size = int(scalar.shape[0])
        for name in metric_names:
            totals[name] += float(losses[name].detach().cpu()) * batch_size
        for name in rate_metric_names:
            totals[f"{name}_num"] += float(losses[f"{name}_num"].detach().cpu())
            totals[f"{name}_den"] += float(losses[f"{name}_den"].detach().cpu())
        totals["samples"] += batch_size
        if train and int(args.step_log_every) > 0 and (step == 1 or step % int(args.step_log_every) == 0):
            samples = max(int(totals["samples"]), 1)
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"step={step:05d}/{len(loader):05d} loss={totals['loss']/samples:.6f} "
                f"xy={totals['xy']/samples:.6f} speed={totals['speed']/samples:.6f} "
                f"smooth={totals['traj_smooth']/samples:.6f}/{totals['speed_smooth']/samples:.6f} "
                f"stopceil={totals['stop_ceiling']/samples:.6f} stop={totals['stop']/samples:.6f} "
                f"risk={totals['behavior_proxy']/samples:.6f} lane={totals['lateral_error']/samples:.6f} "
                f"ctrl={totals['controller_proxy']/samples:.6f} "
                f"plan={totals['plan_steer_error']/samples:.6f}/{totals['plan_throttle_error']/samples:.6f}/{totals['plan_brake_error']/samples:.6f} "
                f"res={totals['output_residual_checkpoint_gate']/samples:.3f}/{totals['output_residual_speed_gate']/samples:.3f}/"
                f"{totals['output_residual_checkpoint_norm']/samples:.3f}/{totals['output_residual_speed_norm']/samples:.3f} "
                f"prior={totals['prior_xy']/samples:.6f}/{totals['prior_speed']/samples:.6f} "
                f"aux={totals['control']/samples:.6f}/{totals['stop_state_aux']/samples:.6f}/{totals['stop_reason_aux']/samples:.6f} "
                f"drift={totals['drift']/samples:.6f} xmodal={totals['xmodal']/samples:.4f} depth={totals['depth']/samples:.4f} moving={totals['moving_ratio']/samples:.3f} "
                f"samples/s={samples/elapsed:.1f}",
                flush=True,
            )
    samples = max(int(totals.pop("samples")), 1)
    result = {name: totals[name] / samples for name in metric_names}
    for name in rate_metric_names:
        result[name] = totals[f"{name}_num"] / max(totals[f"{name}_den"], 1.0)
        result[f"{name}_count"] = totals[f"{name}_den"]
    result["closed_loop_proxy"] = (
        0.35 * result["behavior_proxy"]
        + 0.75 * result["lateral_error"]
        + 0.20 * result["progress_error"]
        + _miss_rate(result, "stop_hold_recall", 2.0)
        + _miss_rate(result, "go_recall", 0.5)
        + _miss_rate(result, "hazard_hold_recall", 3.0)
        + _miss_rate(result, "launch_go_recall", 1.0)
        + _miss_rate(result, "release_go_recall", 1.0)
        + _miss_rate(result, "go_progress_recall", 0.5)
        + _miss_rate(result, "stop_progress_hold", 2.0)
        + _miss_rate(result, "hazard_progress_hold", 3.0)
        + _miss_rate(result, "trafficlight_hold_recall", 2.0)
        + _miss_rate(result, "stopsign_hold_recall", 1.0)
    )
    result["controller_closed_loop_proxy"] = (
        result["closed_loop_proxy"]
        + 2.00 * result["controller_proxy"]
        + 0.75 * result["plan_steer_error"]
        + 0.50 * result["plan_throttle_error"]
        + 2.00 * result["plan_brake_error"]
        + _miss_rate(result, "plan_brake_recall", 3.0)
        + _miss_rate(result, "plan_go_recall", 1.5)
        + _miss_rate(result, "plan_steer_close", 1.0)
        + _miss_rate(result, "plan_throttle_close", 1.0)
    )
    return result


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = _camera_list(args.cameras)
    train_indices, val_indices = split_by_episode(args.index, val_ratio=float(args.val_ratio), seed=int(args.seed))
    if int(args.max_train_samples) > 0:
        train_indices = train_indices[: int(args.max_train_samples)]
    if int(args.max_val_samples) > 0:
        val_indices = val_indices[: int(args.max_val_samples)]

    _teacher_root = args.teacher_view_root or None
    train_ds = Teach2DriveIndexDataset(
        args.index,
        indices=train_indices,
        cameras=cameras,
        image_size=tuple(args.image_size),
        lidar_size=int(args.lidar_size),
        episode_root_override=args.episode_root_override,
        teacher_view_root=_teacher_root,
        teacher_view_dirname=args.teacher_view_dirname,
        teacher_view_camera=args.tfpp_camera,
    )
    val_ds = Teach2DriveIndexDataset(
        args.index,
        indices=val_indices,
        cameras=cameras,
        image_size=tuple(args.image_size),
        lidar_size=int(args.lidar_size),
        episode_root_override=args.episode_root_override,
        teacher_view_root=_teacher_root,
        teacher_view_dirname=args.teacher_view_dirname,
        teacher_view_camera=args.tfpp_camera,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # v5: avoid size-1 last batch breaking BatchNorm train mode
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    net, config, load_info = load_transfuserpp(args.garage_root, args.team_config, device=device, checkpoint=args.checkpoint)
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    peft_lora_modules: list[str] = []
    if int(args.lora_rank) > 0:
        peft_lora_modules = install_lora_adapters(
            net,
            include=args.lora_include,
            exclude=args.lora_exclude,
            rank=int(args.lora_rank),
            alpha=float(args.lora_alpha),
            dropout=float(args.lora_dropout),
        )
        if not peft_lora_modules:
            raise ValueError(f"No LoRA modules matched include={args.lora_include!r} exclude={args.lora_exclude!r}")
    unfrozen_tfpp_params = _unfreeze_matching_tfpp_parameters(net, args.unfreeze_include, args.unfreeze_exclude)

    probe_loader = DataLoader(
        train_ds,
        batch_size=max(1, min(int(args.batch_size), 2)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    stage_feature_shapes, fused_feature_shape = _infer_feature_shapes(net, config, probe_loader, cameras, args, device)
    stage_adapter_layers = _parse_stage_adapter_layers(args.stage_adapter_layers)
    fusion_adapter_enabled = not bool(args.disable_fusion_adapter)
    extrinsic_vector = build_extrinsic_vector(args.source_profile)
    if args.extrinsic_aware:
        adapter = ExtrinsicAwareFeatureThenFusionAdapter(
            stage_feature_shapes=stage_feature_shapes,
            fused_feature_shape=fused_feature_shape,
            extrinsic_vector=extrinsic_vector,
            hidden_channels=int(args.hidden_channels),
            blocks=int(args.blocks),
            dropout=float(args.dropout),
            extrinsic_hidden_dim=int(args.extrinsic_hidden_dim),
            extrinsic_dropout=float(args.extrinsic_dropout),
        ).to(device)
    else:
        adapter = FeatureThenFusionAdapter(
            stage_feature_shapes=stage_feature_shapes,
            fused_feature_shape=fused_feature_shape,
            hidden_channels=int(args.hidden_channels),
            blocks=int(args.blocks),
            dropout=float(args.dropout),
        ).to(device)
    load_adapter_info = load_feature_then_fusion_checkpoint(adapter, args.init_checkpoint, strict=False)
    model = _TaskAdapterForward(
        net=net,
        adapter=adapter,
        config=config,
        cameras=cameras,
        command_mode=args.command_mode,
        tfpp_camera=args.tfpp_camera,
        stage_feature_shapes=stage_feature_shapes,
        fused_feature_shape=fused_feature_shape,
        stage_blend=float(args.stage_feature_adapter_blend),
        fusion_blend=float(args.fusion_adapter_blend),
        stage_adapter_layers=stage_adapter_layers,
        stage_adapter_modalities=str(args.stage_adapter_modalities),
        fusion_adapter_enabled=fusion_adapter_enabled,
        aux_hidden_dim=int(args.aux_hidden_dim),
        use_stop_state_aux=float(args.stop_state_aux_loss_weight) > 0.0,
        use_stop_reason_aux=float(args.stop_reason_aux_loss_weight) > 0.0,
        use_control_aux=float(args.control_loss_weight) > 0.0,
        use_output_residual=bool(args.output_residual),
        output_residual_hidden_dim=int(args.output_residual_hidden_dim),
        output_residual_checkpoint_scale=float(args.output_residual_checkpoint_scale),
        output_residual_speed_logit_scale=float(args.output_residual_speed_logit_scale),
        output_residual_gate_bias=float(args.output_residual_gate_bias),
        output_residual_dropout=float(args.output_residual_dropout),
        camera_crop_shift_x_px=float(args.camera_crop_shift_x_px),
        camera_crop_shift_y_px=float(args.camera_crop_shift_y_px),
        camera_crop_scale=float(args.camera_crop_scale),
        camera_ground_plane_warp=bool(args.camera_ground_plane_warp),
        camera_ground_plane_source_pose=tuple(float(v) for v in args.camera_ground_plane_source_pose),
        camera_ground_plane_target_pose=tuple(float(v) for v in args.camera_ground_plane_target_pose),
        camera_ground_plane_z_m=float(args.camera_ground_plane_z_m),
        lidar_shift_x_m=float(args.lidar_canonical_shift_x_m),
        lidar_shift_y_m=float(args.lidar_canonical_shift_y_m),
        lidar_pixels_per_meter=float(args.lidar_pixels_per_meter),
    ).to(device)
    init_extra_load_info = _load_task_checkpoint_extras(model, args.init_checkpoint, strict=False)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    main_params = []
    unfrozen_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        bare_name = name[len("module.") :] if name.startswith("module.") else name
        is_tfpp_base = bare_name.startswith("net.") and ".lora_" not in bare_name
        if is_tfpp_base:
            unfrozen_params.append(param)
        else:
            main_params.append(param)
    param_groups = []
    if main_params:
        param_groups.append({"params": main_params, "lr": float(args.lr), "weight_decay": float(args.weight_decay)})
    if unfrozen_params:
        param_groups.append(
            {
                "params": unfrozen_params,
                "lr": float(args.unfreeze_lr) if float(args.unfreeze_lr) > 0.0 else float(args.lr),
                "weight_decay": float(args.unfreeze_weight_decay),
            }
        )
    optimizer = torch.optim.AdamW(param_groups)
    args._trainable_anchor_state = _capture_trainable_anchor(model, float(args.init_param_anchor_loss_weight))

    metadata_args = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
    metadata = {
        "mode": "transfuserpp_extrinsic_task_feature_then_fusion_adapter" if args.extrinsic_aware else "transfuserpp_task_feature_then_fusion_adapter",
        "index": str(Path(args.index).expanduser()),
        "episode_root_override": str(Path(args.episode_root_override).expanduser()) if args.episode_root_override else "",
        "garage_root": str(Path(args.garage_root).expanduser()),
        "team_config": str(Path(args.team_config).expanduser()),
        "tfpp_load_info": load_info,
        "adapter_init_checkpoint": str(Path(args.init_checkpoint).expanduser()) if args.init_checkpoint else "",
        "adapter_init_load_info": load_adapter_info,
        "task_init_extra_load_info": init_extra_load_info,
        "init_param_anchor_loss_weight": float(args.init_param_anchor_loss_weight),
        "peft_lora": {
            "rank": int(args.lora_rank),
            "alpha": float(args.lora_alpha),
            "dropout": float(args.lora_dropout),
            "include": args.lora_include,
            "exclude": args.lora_exclude,
            "modules": list(peft_lora_modules),
        },
        "unfrozen_tfpp": {
            "include": args.unfreeze_include,
            "exclude": args.unfreeze_exclude,
            "lr": float(args.unfreeze_lr) if float(args.unfreeze_lr) > 0.0 else float(args.lr),
            "weight_decay": float(args.unfreeze_weight_decay),
            "count": int(len(unfrozen_tfpp_params)),
            "preview": list(unfrozen_tfpp_params[:80]),
        },
        "cameras": cameras,
        "tfpp_camera": args.tfpp_camera,
        "command_mode": args.command_mode,
        "image_size": list(args.image_size),
        "camera_canonical_crop": {
            "shift_x_px": float(args.camera_crop_shift_x_px),
            "shift_y_px": float(args.camera_crop_shift_y_px),
            "scale": float(args.camera_crop_scale),
        },
        "camera_ground_plane_warp": {
            "enabled": bool(args.camera_ground_plane_warp),
            "source_pose": [float(v) for v in args.camera_ground_plane_source_pose],
            "target_pose": [float(v) for v in args.camera_ground_plane_target_pose],
            "ground_z_m": float(args.camera_ground_plane_z_m),
        },
        "lidar_size": int(args.lidar_size),
        "lidar_canonical_shift_m": [float(args.lidar_canonical_shift_x_m), float(args.lidar_canonical_shift_y_m)],
        "lidar_pixels_per_meter": float(args.lidar_pixels_per_meter),
        "stage_feature_shapes": {key: list(value) for key, value in stage_feature_shapes.items()},
        "fused_feature_shape": list(fused_feature_shape),
        "stage_adapter_layers": None if stage_adapter_layers is None else [int(v) for v in stage_adapter_layers],
        "stage_adapter_modalities": str(args.stage_adapter_modalities),
        "fusion_adapter_enabled": bool(fusion_adapter_enabled),
        "output_residual": {
            "enabled": bool(args.output_residual),
            "hidden_dim": int(args.output_residual_hidden_dim),
            "checkpoint_scale": float(args.output_residual_checkpoint_scale),
            "speed_logit_scale": float(args.output_residual_speed_logit_scale),
            "gate_bias": float(args.output_residual_gate_bias),
            "dropout": float(args.output_residual_dropout),
            "checkpoint_dim": int((getattr(config, "predict_checkpoint_len", 10) or 10) * 2),
            "speed_classes": int(len(getattr(config, "target_speeds", [])) or 1),
        },
        "extrinsic_aware": bool(args.extrinsic_aware),
        "source_profile": args.source_profile,
        "extrinsic_vector": [float(v) for v in extrinsic_vector],
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "data_parallel": bool(args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1),
        "args": metadata_args,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": metadata["mode"],
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "stage_feature_shapes": metadata["stage_feature_shapes"],
                "fused_feature_shape": metadata["fused_feature_shape"],
                "stage_adapter_layers": metadata["stage_adapter_layers"],
                "stage_adapter_modalities": metadata["stage_adapter_modalities"],
                "fusion_adapter_enabled": metadata["fusion_adapter_enabled"],
                "output_residual": metadata["output_residual"],
                "tfpp_load_info": load_info,
                "adapter_init_load_info": load_adapter_info,
                "task_init_extra_load_info": init_extra_load_info,
                "peft_lora_modules": peft_lora_modules,
                "unfrozen_tfpp_count": len(unfrozen_tfpp_params),
                "unfrozen_tfpp_preview": unfrozen_tfpp_params[:20],
                "camera_canonical_crop": metadata["camera_canonical_crop"],
                "camera_ground_plane_warp": metadata["camera_ground_plane_warp"],
                "lidar_canonical_shift_m": metadata["lidar_canonical_shift_m"],
                "data_parallel": metadata["data_parallel"],
            },
            indent=2,
        ),
        flush=True,
    )

    if args.selection_metric == "loss":
        selection_name = "loss"
    else:
        selection_name = str(args.selection_metric)
    selection_mode = str(args.selection_mode)
    best_score = float("-inf") if selection_mode == "max" else float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    epoch_checkpoint_dir = out_dir / str(args.epoch_checkpoint_dir)
    if bool(args.save_epoch_checkpoints):
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _cpu_state(sd: dict) -> dict:
        # Pickling large GPU state dicts has intermittently raised a torch internal error
        # during torch.save; moving to CPU first avoids it and is cheaper to serialize.
        return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}

    def save_task_checkpoint(path: Path, epoch: int, train_metrics: dict, val_metrics: dict, selection_value: float) -> None:
        raw_model = _raw_task_model(model)
        payload = {
            "model_state": _cpu_state(raw_model.adapter.state_dict()),
            "peft_lora_state": _cpu_state(lora_state_dict(raw_model.net)) if int(args.lora_rank) > 0 else {},
            # v5: persist the FULL TF++ net (trained/unfrozen backbone + adapted BatchNorm
            # running stats + LoRA base), so backbone-unfreeze and BN adaptation actually
            # transfer to closed-loop eval instead of reverting to the frozen base weights.
            "tfpp_state": _cpu_state(raw_model.net.state_dict()),
            "aux_state": raw_model.aux_state_dict(),
            "stage_feature_shapes": stage_feature_shapes,
            "fused_feature_shape": fused_feature_shape,
            "metadata": metadata,
            "epoch": int(epoch),
            "selection_metric": selection_name,
            "selection_mode": selection_mode,
            "selection_value": float(selection_value),
            "val_metrics": val_metrics,
            "train_metrics": train_metrics,
        }
        # Atomic + fault-tolerant: a transient save failure must not kill the whole run.
        tmp = Path(str(path) + ".tmp")
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] checkpoint save failed for {path} (epoch {epoch}): {exc!r} -- continuing", flush=True)
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    try:
        for epoch in range(1, int(args.epochs) + 1):
            train_metrics = _run_epoch(model, train_loader, optimizer, device, args, train=True, epoch=epoch)
            val_metrics = _run_epoch(model, val_loader, optimizer, device, args, train=False, epoch=epoch)
            # v9/v10: off-pose (perturbed) validation metric -- predicts closed-loop where expert-pose
            # metrics fail. Enabled by T2D_PERTURB_VAL=1 (selection metric only, independent of whether
            # training uses perturbation) OR whenever recovery-perturbation training is on.
            if os.environ.get("T2D_PERTURB_VAL", "0") == "1" or float(_recovery_perturb_config()["prob"]) > 0.0:
                _cpu_rng = torch.get_rng_state()
                _cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                pv = _run_epoch(model, val_loader, optimizer, device, args, train=False, epoch=epoch, force_perturb=True)
                torch.set_rng_state(_cpu_rng)
                if _cuda_rng is not None:
                    torch.cuda.set_rng_state_all(_cuda_rng)
                val_metrics["perturbed_loss"] = pv["loss"]
                val_metrics["perturbed_xy"] = pv["xy"]
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            val_loss = float(val_metrics["loss"])
            if selection_name not in val_metrics:
                raise KeyError(f"selection metric {selection_name!r} not in validation metrics")
            selection_value = float(val_metrics[selection_name])
            if selection_mode == "max":
                improved = selection_value > best_score + float(args.early_stop_min_delta)
            else:
                improved = selection_value + float(args.early_stop_min_delta) < best_score
            display_best = selection_value if improved or not best_epoch else best_score
            print(
                f"epoch={epoch:03d} train={train_metrics['loss']:.6f} val={val_loss:.6f} "
                f"select={selection_name}:{selection_value:.6f} best={display_best:.6f} new_best={int(improved)} "
                f"xy={val_metrics['xy']:.6f} speed={val_metrics['speed']:.6f} "
                f"closed={val_metrics['closed_loop_proxy']:.6f} risk={val_metrics['behavior_proxy']:.6f} "
                f"lane={val_metrics['lateral_error']:.6f} prog={val_metrics['progress_error']:.6f} "
                f"ctrl_closed={val_metrics['controller_closed_loop_proxy']:.6f} "
                f"ctrl={val_metrics['controller_proxy']:.6f} "
                f"plan={val_metrics['plan_steer_error']:.6f}/{val_metrics['plan_throttle_error']:.6f}/{val_metrics['plan_brake_error']:.6f} "
                f"sg={val_metrics['stop_hold_recall']:.3f}/{val_metrics['go_recall']:.3f} "
                f"hz={val_metrics['hazard_hold_recall']:.3f} "
                f"plan_sgtb={val_metrics['plan_steer_close']:.3f}/{val_metrics['plan_go_recall']:.3f}/"
                f"{val_metrics['plan_throttle_close']:.3f}/{val_metrics['plan_brake_recall']:.3f} "
                f"lr={val_metrics['launch_go_recall']:.3f}/{val_metrics['release_go_recall']:.3f} "
                f"res={val_metrics['output_residual_checkpoint_gate']:.3f}/{val_metrics['output_residual_speed_gate']:.3f}/"
                f"{val_metrics['output_residual_checkpoint_norm']:.3f}/{val_metrics['output_residual_speed_norm']:.3f} "
                f"prior={val_metrics['prior_xy']:.6f}/{val_metrics['prior_speed']:.6f} "
                f"aux={val_metrics['control']:.6f}/{val_metrics['stop_state_aux']:.6f}/{val_metrics['stop_reason_aux']:.6f} "
                f"drift={val_metrics['drift']:.6f} xmodal={val_metrics.get('xmodal', 0.0):.4f}",
                flush=True,
            )
            if "perturbed_loss" in val_metrics:
                print(
                    f"  [offpose] perturbed_loss={float(val_metrics['perturbed_loss']):.6f} "
                    f"perturbed_xy={float(val_metrics['perturbed_xy']):.6f} "
                    f"(clean val_loss={val_loss:.6f} xy={val_metrics['xy']:.6f})",
                    flush=True,
                )
            if bool(args.save_epoch_checkpoints):
                epoch_path = epoch_checkpoint_dir / f"epoch_{epoch:03d}.pt"
                save_task_checkpoint(epoch_path, epoch, train_metrics, val_metrics, selection_value)
            if improved:
                best_score = selection_value
                best_val_loss = val_loss
                best_epoch = epoch
                stale = 0
                save_task_checkpoint(out_dir / "best_model.pt", epoch, train_metrics, val_metrics, selection_value)
            else:
                stale += 1
                if stale >= int(args.early_stop_patience):
                    print(
                        f"early_stop: no val improvement for {stale} epochs "
                        f"(patience={args.early_stop_patience}, {selection_name}={best_score:.6f})",
                        flush=True,
                    )
                    break
    finally:
        _raw_task_model(model).restore()

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_selection_metric": selection_name,
        "best_selection_mode": selection_mode,
        "best_selection_value": float(best_score),
        "mode": metadata["mode"],
    }
    if (out_dir / "best_model.pt").exists():
        best = torch.load(out_dir / "best_model.pt", map_location="cpu")
        summary.update(best.get("val_metrics", {}))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a target-only task-driven TransFuser++ feature-then-fusion adapter.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--episode-root-override", default="")
    # v4 geometric-teacher distillation: root that holds <source_route>/<dirname>/<step>.jpg
    # reprojected x=-1.5 views. Empty -> disabled (drift stays a self-reference regularizer).
    parser.add_argument("--teacher-view-root", default="")
    parser.add_argument("--teacher-view-dirname", default="rgb_front_teacher_xm15")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--garage-root", required=True)
    parser.add_argument("--team-config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--init-param-anchor-loss-weight", type=float, default=0.0)
    parser.add_argument("--cameras", default="left,front,right")
    parser.add_argument("--tfpp-camera", default="front")
    parser.add_argument("--command-mode", choices=["lane_follow", "target_angle"], default="target_angle")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--camera-crop-shift-x-px", type=float, default=0.0)
    parser.add_argument("--camera-crop-shift-y-px", type=float, default=0.0)
    parser.add_argument("--camera-crop-scale", type=float, default=1.0)
    parser.add_argument("--camera-ground-plane-warp", action="store_true")
    parser.add_argument(
        "--camera-ground-plane-source-pose",
        type=float,
        nargs=6,
        default=[1.25, 0.0, 1.95, 0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
    )
    parser.add_argument(
        "--camera-ground-plane-target-pose",
        type=float,
        nargs=6,
        default=[-1.5, 0.0, 2.0, 0.0, 0.0, 0.0],
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
    )
    parser.add_argument("--camera-ground-plane-z-m", type=float, default=0.0)
    parser.add_argument("--lidar-size", type=int, default=128)
    parser.add_argument("--extrinsic-aware", action="store_true")
    parser.add_argument("--source-profile", default="front_triplet_shifted", choices=["front_triplet_shifted", "tfpp_ego"])
    parser.add_argument("--extrinsic-hidden-dim", type=int, default=64)
    parser.add_argument("--extrinsic-dropout", type=float, default=0.0)
    parser.add_argument("--hidden-channels", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--stage-adapter-layers",
        default="all",
        help="Stage fusion layers to adapt: all, early:N, or comma-separated indices such as 0,1.",
    )
    parser.add_argument(
        "--stage-adapter-modalities",
        choices=["all", "camera", "lidar", "none"],
        default="all",
        help="Which stage token streams are replaced by the learned adapter output.",
    )
    parser.add_argument("--disable-fusion-adapter", action="store_true")
    parser.add_argument("--stage-feature-adapter-blend", type=float, default=1.0)
    parser.add_argument("--fusion-adapter-blend", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-include",
        default="^join\\.,^checkpoint_decoder\\.(encoder|decoder)\\.,^target_speed_network\\.",
    )
    parser.add_argument("--lora-exclude", default="")
    parser.add_argument(
        "--unfreeze-include",
        default="",
        help="Comma-separated regexes for TF++ base parameters to fine-tune, e.g. '^backbone\\.(image_encoder|lidar_encoder)\\.(stem|s1|s2)'.",
    )
    parser.add_argument("--unfreeze-exclude", default="")
    parser.add_argument("--unfreeze-lr", type=float, default=0.0)
    parser.add_argument("--unfreeze-weight-decay", type=float, default=1e-5)
    parser.add_argument("--lidar-canonical-shift-x-m", type=float, default=0.0)
    parser.add_argument("--lidar-canonical-shift-y-m", type=float, default=0.0)
    parser.add_argument("--lidar-pixels-per-meter", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--selection-metric", default="loss")
    parser.add_argument("--selection-mode", choices=["min", "max"], default="min")
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--epoch-checkpoint-dir", default="epoch_checkpoints")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--speed-dim", type=int, default=4)
    parser.add_argument("--xy-loss-weight", type=float, default=0.55)
    parser.add_argument("--yaw-loss-weight", type=float, default=0.03)
    parser.add_argument("--speed-loss-weight", type=float, default=0.80)
    parser.add_argument("--traj-smooth-loss-weight", type=float, default=0.0)
    parser.add_argument("--speed-smooth-loss-weight", type=float, default=0.0)
    parser.add_argument("--speed-floor-loss-weight", type=float, default=0.03)
    parser.add_argument("--speed-floor-mps", type=float, default=0.8)
    parser.add_argument("--speed-floor-target-threshold", type=float, default=2.0)
    parser.add_argument("--stop-speed-ceiling-loss-weight", type=float, default=0.35)
    parser.add_argument("--stop-speed-ceiling-mps", type=float, default=0.5)
    parser.add_argument("--stop-speed-target-threshold", type=float, default=0.5)
    parser.add_argument("--stop-progress-ceiling-m", type=float, default=1.0)
    parser.add_argument("--go-progress-ratio", type=float, default=0.5)
    parser.add_argument("--stop-loss-weight", type=float, default=0.08)
    parser.add_argument("--feature-drift-loss-weight", type=float, default=0.10)
    parser.add_argument("--output-prior-xy-loss-weight", type=float, default=0.0)
    parser.add_argument("--output-prior-speed-loss-weight", type=float, default=0.0)
    parser.add_argument("--aux-hidden-dim", type=int, default=256)
    parser.add_argument("--control-loss-weight", type=float, default=0.0)
    parser.add_argument("--pdm-behavior-loss-weight", type=float, default=0.0)
    parser.add_argument("--pdm-lateral-loss-weight", type=float, default=0.0)
    parser.add_argument("--pdm-progress-loss-weight", type=float, default=0.0)
    parser.add_argument("--pdm-hazard-progress-loss-weight", type=float, default=0.0)
    parser.add_argument("--pdm-controller-loss-weight", type=float, default=0.0)
    parser.add_argument("--pdm-plan-steer-loss-weight", type=float, default=0.0)
    parser.add_argument("--pdm-plan-throttle-loss-weight", type=float, default=0.0)
    parser.add_argument("--pdm-plan-brake-loss-weight", type=float, default=0.0)
    parser.add_argument("--output-residual", action="store_true")
    parser.add_argument("--output-residual-hidden-dim", type=int, default=256)
    parser.add_argument("--output-residual-checkpoint-scale", type=float, default=0.75)
    parser.add_argument("--output-residual-speed-logit-scale", type=float, default=1.5)
    parser.add_argument("--output-residual-gate-bias", type=float, default=-2.0)
    parser.add_argument("--output-residual-dropout", type=float, default=0.0)
    parser.add_argument("--stop-state-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--stop-reason-aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--moving-speed-threshold", type=float, default=1.0)
    parser.add_argument("--moving-sample-weight", type=float, default=1.15)
    parser.add_argument("--stopped-sample-weight", type=float, default=1.0)
    parser.add_argument("--hazard-stop-reasons", default="traffic_light,stop_sign,front_vehicle,junction_yield")
    parser.add_argument("--hazard-sample-weight", type=float, default=2.0)
    parser.add_argument("--launch-current-speed-threshold", type=float, default=0.8)
    parser.add_argument("--launch-target-speed-threshold", type=float, default=2.0)
    parser.add_argument("--launch-sample-weight", type=float, default=1.4)
    parser.add_argument("--launch-speed-floor-loss-weight", type=float, default=0.04)
    parser.add_argument("--launch-speed-floor-mps", type=float, default=1.2)
    parser.add_argument("--release-target-speed-threshold", type=float, default=1.0)
    parser.add_argument("--release-sample-weight", type=float, default=1.3)
    parser.add_argument("--release-speed-floor-loss-weight", type=float, default=0.04)
    parser.add_argument("--release-speed-floor-mps", type=float, default=1.2)
    parser.add_argument("--controller-steer-close-threshold", type=float, default=0.15)
    parser.add_argument("--controller-throttle-close-threshold", type=float, default=0.20)
    parser.add_argument("--controller-go-throttle-threshold", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--step-log-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
