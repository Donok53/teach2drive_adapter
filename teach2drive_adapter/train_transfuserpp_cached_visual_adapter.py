import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import STOP_REASON_NAMES, STOP_STATE_NAMES, Teach2DriveIndexDataset
from .layout_conditioning import FiLMLayoutAdapter
from .model import ConvEncoder, count_trainable_parameters
from .train_adapter import _masked_weighted_ce, _weighted_mean
from .train_transfuserpp_cached_adapter import _split_by_episode
from .transfuserpp_adapter_model import TransFuserPPResidualHeads


CAMERA_LAYOUT_ORDER = ("left", "front", "right")
CAMERA_LAYOUT_WIDTH = 13
LIDAR_LAYOUT_START = len(CAMERA_LAYOUT_ORDER) * CAMERA_LAYOUT_WIDTH
LIDAR_LAYOUT_WIDTH = 12


def _camera_list(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def _metadata_from_cache(arrays) -> Dict:
    if "metadata" not in arrays.files:
        return {}
    return json.loads(str(arrays["metadata"].item()))


def _resolve_from_metadata(value: str, metadata: Dict, key: str) -> str:
    if value:
        return value
    return str(metadata.get(key, ""))


class CachedVisualPriorDataset(Dataset):
    """Cached TransFuser++ prior plus raw Teach2Drive camera/LiDAR tensors.

    The cache stores frozen TransFuser++ outputs. This dataset reopens the
    original Teach2Drive index by cached sample ids so the adapter can still
    use the full sensor observations without rerunning TransFuser++.
    """

    def __init__(
        self,
        cache_path: str,
        index_path: str,
        episode_root_override: str,
        indices: Optional[np.ndarray] = None,
        cameras: Optional[Sequence[str]] = None,
        image_size=(320, 180),
        lidar_size: int = 128,
        use_raw_layout: bool = False,
        teacher_cache_path: str = "",
    ) -> None:
        self.cache_path = Path(cache_path).expanduser()
        arrays = np.load(self.cache_path, allow_pickle=True)
        self.metadata = _metadata_from_cache(arrays)
        self.sample_index = arrays["sample_index"].astype(np.int64)
        self.sample_episode = arrays["sample_episode"].astype(np.int64)
        self.sample_frame = arrays["sample_frame"].astype(np.int64)
        self.scalar = arrays["scalar"].astype(np.float32)
        self.layout = arrays["layout"].astype(np.float32)
        self.target = arrays["target"].astype(np.float32)
        self.stop_state = arrays["stop_state"].astype(np.int64)
        self.stop_reason = arrays["stop_reason"].astype(np.int64)
        self.stop_reason_mask = arrays["stop_reason_mask"].astype(np.float32)
        self.sample_weight = arrays["sample_weight"].astype(np.float32)
        self.base_target = arrays["base_target"].astype(np.float32)
        self.checkpoint_flat = arrays["checkpoint_flat"].astype(np.float32)
        self.speed_logits = arrays["speed_logits"].astype(np.float32)
        self.expected_speed = arrays["expected_speed"].astype(np.float32)
        self.indices = np.arange(len(self.sample_index), dtype=np.int64) if indices is None else indices.astype(np.int64)
        self.use_raw_layout = bool(use_raw_layout)
        self.teacher_cache_path = str(Path(teacher_cache_path).expanduser()) if teacher_cache_path else ""
        self.teacher_metadata: Dict = {}
        self.teacher_target = None
        self.teacher_checkpoint_flat = None
        self.teacher_speed_logits = None
        self.teacher_expected_speed = None
        if self.teacher_cache_path:
            teacher_arrays = np.load(self.teacher_cache_path, allow_pickle=True)
            self.teacher_metadata = _metadata_from_cache(teacher_arrays)
            teacher_episode = teacher_arrays["sample_episode"].astype(np.int64)
            teacher_frame = teacher_arrays["sample_frame"].astype(np.int64)
            if len(teacher_episode) != len(self.sample_episode):
                raise ValueError(f"teacher cache length mismatch: {len(teacher_episode)} != {len(self.sample_episode)}")
            if not np.array_equal(teacher_episode, self.sample_episode) or not np.array_equal(teacher_frame, self.sample_frame):
                raise ValueError("teacher cache is not aligned with the input cache by episode/frame")
            self.teacher_target = teacher_arrays["base_target"].astype(np.float32)
            self.teacher_checkpoint_flat = teacher_arrays["checkpoint_flat"].astype(np.float32)
            self.teacher_speed_logits = teacher_arrays["speed_logits"].astype(np.float32)
            self.teacher_expected_speed = teacher_arrays["expected_speed"].astype(np.float32)

        raw_sample_indices = self.sample_index[self.indices]
        self.raw = Teach2DriveIndexDataset(
            index_path,
            indices=raw_sample_indices,
            cameras=cameras,
            image_size=tuple(int(v) for v in image_size),
            lidar_size=int(lidar_size),
            episode_root_override=episode_root_override,
        )

    @property
    def scalar_dim(self) -> int:
        return int(self.scalar.shape[1])

    @property
    def layout_dim(self) -> int:
        return int(self.layout.shape[1])

    @property
    def target_dim(self) -> int:
        return int(self.target.shape[1])

    @property
    def camera_count(self) -> int:
        return len(self.raw.cameras)

    @property
    def lidar_channels(self) -> int:
        if len(self.raw) == 0:
            return 1
        return int(self.raw[0]["lidar"].shape[0])

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        cache_idx = int(self.indices[item])
        raw = self.raw[item]
        layout = raw["layout"] if self.use_raw_layout else torch.from_numpy(self.layout[cache_idx])
        return {
            "camera": raw["camera"],
            "lidar": raw["lidar"],
            "scalar": torch.from_numpy(self.scalar[cache_idx]),
            "layout": layout,
            "target": torch.from_numpy(self.target[cache_idx]),
            "stop_state": raw["stop_state"],
            "stop_reason": raw["stop_reason"],
            "stop_reason_mask": raw["stop_reason_mask"],
            "sample_weight": raw["sample_weight"],
            "base_target": torch.from_numpy(self.base_target[cache_idx]),
            "checkpoint_flat": torch.from_numpy(self.checkpoint_flat[cache_idx]),
            "speed_logits": torch.from_numpy(self.speed_logits[cache_idx]),
            "expected_speed": torch.from_numpy(self.expected_speed[cache_idx]),
            "teacher_target": torch.from_numpy(self.teacher_target[cache_idx]) if self.teacher_target is not None else torch.from_numpy(self.target[cache_idx]),
            "teacher_checkpoint_flat": torch.from_numpy(self.teacher_checkpoint_flat[cache_idx]) if self.teacher_checkpoint_flat is not None else torch.from_numpy(self.checkpoint_flat[cache_idx]),
            "teacher_speed_logits": torch.from_numpy(self.teacher_speed_logits[cache_idx]) if self.teacher_speed_logits is not None else torch.from_numpy(self.speed_logits[cache_idx]),
            "teacher_expected_speed": torch.from_numpy(self.teacher_expected_speed[cache_idx]) if self.teacher_expected_speed is not None else torch.from_numpy(self.expected_speed[cache_idx]),
        }


class MultiSensorVisualEncoder(nn.Module):
    """Encode three-camera observations and LiDAR BEV into adapter features.

    LiDAR is already ego-BEV in the Teach2Drive dataset. Camera features are
    layout-conditioned before fusion, so the adapter can learn how each
    camera pose contributes relative to that BEV anchor.
    """

    def __init__(
        self,
        cameras: Sequence[str],
        lidar_channels: int,
        visual_dim: int = 256,
        token_dim: int = 192,
        transformer_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.cameras = tuple(cameras)
        self.image_encoder = ConvEncoder(3, token_dim)
        self.lidar_encoder = ConvEncoder(lidar_channels, token_dim)
        self.camera_layout = nn.Sequential(
            nn.LayerNorm(CAMERA_LAYOUT_WIDTH),
            nn.Linear(CAMERA_LAYOUT_WIDTH, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.lidar_layout = nn.Sequential(
            nn.LayerNorm(LIDAR_LAYOUT_WIDTH),
            nn.Linear(LIDAR_LAYOUT_WIDTH, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.norm = nn.LayerNorm(token_dim)
        self.out = nn.Sequential(nn.Linear(token_dim, visual_dim), nn.GELU(), nn.LayerNorm(visual_dim))

    def _camera_layout_tokens(self, layout: torch.Tensor) -> torch.Tensor:
        tokens = []
        for camera in self.cameras:
            if camera in CAMERA_LAYOUT_ORDER:
                idx = CAMERA_LAYOUT_ORDER.index(camera)
                start = idx * CAMERA_LAYOUT_WIDTH
                tokens.append(layout[:, start : start + CAMERA_LAYOUT_WIDTH])
            else:
                tokens.append(torch.zeros((layout.shape[0], CAMERA_LAYOUT_WIDTH), dtype=layout.dtype, device=layout.device))
        return torch.stack(tokens, dim=1)

    def _lidar_layout_token(self, layout: torch.Tensor) -> torch.Tensor:
        end = LIDAR_LAYOUT_START + LIDAR_LAYOUT_WIDTH
        if layout.shape[1] >= end:
            return layout[:, LIDAR_LAYOUT_START:end]
        return torch.zeros((layout.shape[0], LIDAR_LAYOUT_WIDTH), dtype=layout.dtype, device=layout.device)

    def forward(self, camera: torch.Tensor, lidar: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        batch, camera_count, channels, height, width = camera.shape
        image = camera.reshape(batch * camera_count, channels, height, width)
        image_feat = self.image_encoder(image).reshape(batch, camera_count, -1)
        camera_layout = self.camera_layout(self._camera_layout_tokens(layout))
        image_feat = image_feat + camera_layout

        lidar_feat = self.lidar_encoder(lidar).unsqueeze(1)
        lidar_feat = lidar_feat + self.lidar_layout(self._lidar_layout_token(layout)).unsqueeze(1)
        tokens = torch.cat([image_feat, lidar_feat], dim=1)
        fused = self.fusion(tokens)
        return self.out(self.norm(fused.mean(dim=1)))


class CachedVisualTransFuserPPAdapterPolicy(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        layout_dim: int,
        target_dim: int,
        checkpoint_dim: int,
        speed_classes: int,
        cameras: Sequence[str],
        lidar_channels: int,
        hidden_dim: int = 512,
        layout_hidden_dim: int = 128,
        visual_dim: int = 256,
        visual_token_dim: int = 192,
        visual_layers: int = 2,
        visual_heads: int = 4,
    ) -> None:
        super().__init__()
        self.visual = MultiSensorVisualEncoder(
            cameras=cameras,
            lidar_channels=lidar_channels,
            visual_dim=visual_dim,
            token_dim=visual_token_dim,
            transformer_layers=visual_layers,
            num_heads=visual_heads,
        )
        self.feature_dim = scalar_dim + layout_dim + target_dim + checkpoint_dim + speed_classes + 1 + visual_dim
        self.layout_adapter = FiLMLayoutAdapter(feature_dim=self.feature_dim, layout_dim=layout_dim, hidden_dim=layout_hidden_dim)
        self.heads = TransFuserPPResidualHeads(self.feature_dim, target_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        layout: torch.Tensor,
        base_target: torch.Tensor,
        checkpoint_flat: torch.Tensor,
        speed_logits: torch.Tensor,
        expected_speed: torch.Tensor,
        camera: torch.Tensor,
        lidar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        visual = self.visual(camera, lidar, layout)
        features = torch.cat([scalar, layout, base_target, checkpoint_flat, speed_logits, expected_speed, visual], dim=1)
        adapted = self.layout_adapter(features, layout)["features"]
        return self.heads(adapted, base_target)


def _speed_region(pred: torch.Tensor, speed_dim: int):
    traj_dim = pred.shape[1] - int(speed_dim) - 1
    return traj_dim, pred[:, traj_dim : traj_dim + int(speed_dim)]


def _moving_mask(scalar: torch.Tensor, target: torch.Tensor, base_target: torch.Tensor, speed_dim: int, threshold: float) -> torch.Tensor:
    traj_dim, target_speed = _speed_region(target, speed_dim)
    base_speed = base_target[:, traj_dim : traj_dim + int(speed_dim)]
    current_speed = scalar[:, :1].abs() if scalar.shape[1] else torch.zeros_like(target_speed[:, :1])
    target_speed_max = target_speed.abs().amax(dim=1, keepdim=True)
    base_speed_max = base_speed.abs().amax(dim=1, keepdim=True)
    return ((target_speed_max > threshold) | (base_speed_max > threshold) | (current_speed > threshold)).reshape(-1)


def _effective_weight(weight: torch.Tensor, moving: torch.Tensor, args) -> torch.Tensor:
    weight = weight.reshape(-1)
    moving = moving.reshape(-1).bool()
    moving_scale = torch.where(
        moving,
        torch.full_like(weight, float(args.moving_sample_weight)),
        torch.full_like(weight, float(args.stopped_sample_weight)),
    )
    return weight * moving_scale


def _stop_loss_weight(target_stop: torch.Tensor, args) -> torch.Tensor:
    return torch.where(
        target_stop >= 0.5,
        torch.full_like(target_stop, float(args.stop_positive_loss_scale)),
        torch.full_like(target_stop, float(args.stop_negative_loss_scale)),
    )


def _run_epoch(model, loader, optimizer, device, args, train: bool, epoch: int = 1) -> Dict[str, float]:
    model.train(train)
    totals = {
        "loss": 0.0,
        "xy": 0.0,
        "yaw": 0.0,
        "speed": 0.0,
        "speed_distill": 0.0,
        "speed_floor": 0.0,
        "speed_delta": 0.0,
        "speed_curvature": 0.0,
        "traj_delta": 0.0,
        "traj_curvature": 0.0,
        "prior": 0.0,
        "stop": 0.0,
        "state": 0.0,
        "reason": 0.0,
        "moving_ratio": 0.0,
        "samples": 0,
    }
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        scalar = batch["scalar"].to(device, non_blocking=True)
        layout = batch["layout"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        teacher_target = batch["teacher_target"].to(device, non_blocking=True)
        base_target = batch["base_target"].to(device, non_blocking=True)
        checkpoint_flat = batch["checkpoint_flat"].to(device, non_blocking=True)
        speed_logits = batch["speed_logits"].to(device, non_blocking=True)
        expected_speed = batch["expected_speed"].to(device, non_blocking=True)
        camera = batch["camera"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        stop_state = batch["stop_state"].to(device, non_blocking=True)
        stop_reason = batch["stop_reason"].to(device, non_blocking=True)
        stop_reason_mask = batch["stop_reason_mask"].to(device, non_blocking=True)
        weight = batch["sample_weight"].to(device, non_blocking=True)
        speed_dim = int(args.speed_dim)
        traj_dim = target.shape[1] - speed_dim - 1
        fallback_blend = float(args.teacher_target_blend)
        traj_blend = fallback_blend if args.teacher_traj_blend is None else float(args.teacher_traj_blend)
        speed_target_blend = fallback_blend if args.teacher_speed_target_blend is None else float(args.teacher_speed_target_blend)
        stop_target_blend = fallback_blend if args.teacher_stop_target_blend is None else float(args.teacher_stop_target_blend)
        target_traj_flat = (
            (1.0 - traj_blend) * target[:, :traj_dim]
            + traj_blend * teacher_target[:, :traj_dim]
        )
        speed_supervision = (
            (1.0 - speed_target_blend) * target[:, traj_dim : traj_dim + speed_dim]
            + speed_target_blend * teacher_target[:, traj_dim : traj_dim + speed_dim]
        )
        stop_supervision = (
            (1.0 - stop_target_blend) * target[:, -1:]
            + stop_target_blend * teacher_target[:, -1:]
        )
        supervision_target = torch.cat([target_traj_flat, speed_supervision, stop_supervision], dim=1)
        moving = _moving_mask(scalar, supervision_target, base_target, speed_dim, float(args.moving_speed_threshold))
        effective_weight = _effective_weight(weight, moving, args)
        with torch.set_grad_enabled(train):
            out = model(scalar, layout, base_target, checkpoint_flat, speed_logits, expected_speed, camera, lidar)
            pred = out["target"]
            pred_traj = pred[:, :traj_dim].reshape(pred.shape[0], -1, 3)
            target_traj = target_traj_flat.reshape(target_traj_flat.shape[0], -1, 3)
            xy_loss = _weighted_mean(torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., :2], target_traj[..., :2], reduction="none"), dim=(1, 2)), effective_weight)
            yaw_loss = _weighted_mean(torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., 2], target_traj[..., 2], reduction="none"), dim=1), effective_weight)
            zero = pred.new_tensor(0.0)
            traj_delta_loss = zero
            traj_curvature_loss = zero
            if pred_traj.shape[1] > 1:
                pred_delta = pred_traj[:, 1:, :2] - pred_traj[:, :-1, :2]
                target_delta = target_traj[:, 1:, :2] - target_traj[:, :-1, :2]
                traj_delta_loss = _weighted_mean(
                    torch.mean(nn.functional.smooth_l1_loss(pred_delta, target_delta, reduction="none"), dim=(1, 2)),
                    effective_weight,
                )
            if pred_traj.shape[1] > 2:
                pred_curvature = pred_traj[:, 2:, :2] - 2.0 * pred_traj[:, 1:-1, :2] + pred_traj[:, :-2, :2]
                target_curvature = target_traj[:, 2:, :2] - 2.0 * target_traj[:, 1:-1, :2] + target_traj[:, :-2, :2]
                traj_curvature_loss = _weighted_mean(
                    torch.mean(nn.functional.smooth_l1_loss(pred_curvature, target_curvature, reduction="none"), dim=(1, 2)),
                    effective_weight,
                )
            pred_speed = pred[:, traj_dim : traj_dim + speed_dim]
            expert_speed = speed_supervision
            base_speed = base_target[:, traj_dim : traj_dim + speed_dim]
            speed_target = (1.0 - float(args.speed_teacher_blend)) * expert_speed + float(args.speed_teacher_blend) * base_speed
            speed_loss = _weighted_mean(
                torch.mean(nn.functional.smooth_l1_loss(pred_speed, speed_target, reduction="none"), dim=1),
                effective_weight,
            )
            speed_distill_loss = _weighted_mean(
                torch.mean(nn.functional.smooth_l1_loss(pred_speed, base_speed.detach(), reduction="none"), dim=1),
                effective_weight,
            )
            moving_float = moving.to(dtype=pred_speed.dtype)
            speed_floor_raw = torch.relu(float(args.speed_floor_mps) - pred_speed).mean(dim=1) * moving_float
            speed_floor_loss = _weighted_mean(speed_floor_raw, effective_weight)
            speed_delta_loss = zero
            speed_curvature_loss = zero
            if speed_dim > 1:
                pred_speed_delta = pred_speed[:, 1:] - pred_speed[:, :-1]
                target_speed_delta = speed_target[:, 1:] - speed_target[:, :-1]
                speed_delta_loss = _weighted_mean(
                    torch.mean(nn.functional.smooth_l1_loss(pred_speed_delta, target_speed_delta, reduction="none"), dim=1),
                    effective_weight,
                )
            if speed_dim > 2:
                pred_speed_curvature = pred_speed[:, 2:] - 2.0 * pred_speed[:, 1:-1] + pred_speed[:, :-2]
                target_speed_curvature = speed_target[:, 2:] - 2.0 * speed_target[:, 1:-1] + speed_target[:, :-2]
                speed_curvature_loss = _weighted_mean(
                    torch.mean(nn.functional.smooth_l1_loss(pred_speed_curvature, target_speed_curvature, reduction="none"), dim=1),
                    effective_weight,
                )
            stop_raw = nn.functional.binary_cross_entropy_with_logits(pred[:, -1:], stop_supervision, reduction="none")
            stop_loss = _weighted_mean(stop_raw * _stop_loss_weight(stop_supervision, args), effective_weight)
            state_loss = _weighted_mean(nn.functional.cross_entropy(out["stop_state"], stop_state, reduction="none"), effective_weight)
            reason_loss = _masked_weighted_ce(out["stop_reason"], stop_reason, stop_reason_mask, effective_weight)
            prior_loss = _weighted_mean(
                torch.mean(nn.functional.smooth_l1_loss(pred[:, : traj_dim + speed_dim], base_target[:, : traj_dim + speed_dim], reduction="none"), dim=1),
                effective_weight,
            )
            stop_loss_weight = float(args.stop_loss_weight) if int(epoch) >= int(args.stop_loss_after_epoch) else 0.0
            stop_state_loss_weight = float(args.stop_state_loss_weight) if int(epoch) >= int(args.stop_loss_after_epoch) else 0.0
            stop_reason_loss_weight = float(args.stop_reason_loss_weight) if int(epoch) >= int(args.stop_loss_after_epoch) else 0.0
            loss = (
                args.xy_loss_weight * xy_loss
                + args.yaw_loss_weight * yaw_loss
                + args.speed_loss_weight * speed_loss
                + args.speed_distill_loss_weight * speed_distill_loss
                + args.speed_floor_loss_weight * speed_floor_loss
                + args.speed_delta_loss_weight * speed_delta_loss
                + args.speed_curvature_loss_weight * speed_curvature_loss
                + args.traj_delta_loss_weight * traj_delta_loss
                + args.traj_curvature_loss_weight * traj_curvature_loss
                + args.prior_loss_weight * prior_loss
                + stop_loss_weight * stop_loss
                + stop_state_loss_weight * state_loss
                + stop_reason_loss_weight * reason_loss
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        batch_size = int(scalar.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["xy"] += float(xy_loss.detach().cpu()) * batch_size
        totals["yaw"] += float(yaw_loss.detach().cpu()) * batch_size
        totals["speed"] += float(speed_loss.detach().cpu()) * batch_size
        totals["speed_distill"] += float(speed_distill_loss.detach().cpu()) * batch_size
        totals["speed_floor"] += float(speed_floor_loss.detach().cpu()) * batch_size
        totals["speed_delta"] += float(speed_delta_loss.detach().cpu()) * batch_size
        totals["speed_curvature"] += float(speed_curvature_loss.detach().cpu()) * batch_size
        totals["traj_delta"] += float(traj_delta_loss.detach().cpu()) * batch_size
        totals["traj_curvature"] += float(traj_curvature_loss.detach().cpu()) * batch_size
        totals["prior"] += float(prior_loss.detach().cpu()) * batch_size
        totals["stop"] += float(stop_loss.detach().cpu()) * batch_size
        totals["state"] += float(state_loss.detach().cpu()) * batch_size
        totals["reason"] += float(reason_loss.detach().cpu()) * batch_size
        totals["moving_ratio"] += float(moving.float().mean().detach().cpu()) * batch_size
        totals["samples"] += batch_size
        if train and args.step_log_every > 0 and (step == 1 or step % args.step_log_every == 0):
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"step={step:05d}/{len(loader):05d} "
                f"loss={totals['loss']/totals['samples']:.6f} "
                f"xy={totals['xy']/totals['samples']:.6f} "
                f"speed={totals['speed']/totals['samples']:.6f} "
                f"floor={totals['speed_floor']/totals['samples']:.6f} "
                f"sd={totals['speed_delta']/totals['samples']:.6f} "
                f"moving={totals['moving_ratio']/totals['samples']:.3f} "
                f"samples/s={totals['samples']/elapsed:.1f}",
                flush=True,
            )
    samples = max(int(totals.pop("samples")), 1)
    return {key: value / samples for key, value in totals.items()}


def _evaluate_predictions(model, loader, device, speed_dim: int = 4, moving_speed_threshold: float = 1.0) -> Dict:
    model.eval()
    stop_correct = 0
    stop_total = 0
    state_correct = 0
    state_total = 0
    reason_correct = 0
    reason_total = 0
    xy_errors = []
    teacher_xy_errors = []
    speed_errors = []
    teacher_speed_errors = []
    speed_delta_errors = []
    speed_curvature_errors = []
    pred_speeds = []
    target_speeds = []
    teacher_speeds = []
    base_speeds = []
    moving_total = 0
    sample_total = 0
    with torch.no_grad():
        for batch in loader:
            scalar = batch["scalar"].to(device, non_blocking=True)
            layout = batch["layout"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            teacher_target = batch["teacher_target"].to(device, non_blocking=True)
            base_target = batch["base_target"].to(device, non_blocking=True)
            checkpoint_flat = batch["checkpoint_flat"].to(device, non_blocking=True)
            speed_logits = batch["speed_logits"].to(device, non_blocking=True)
            expected_speed = batch["expected_speed"].to(device, non_blocking=True)
            camera = batch["camera"].to(device, non_blocking=True)
            lidar = batch["lidar"].to(device, non_blocking=True)
            stop_state = batch["stop_state"].to(device, non_blocking=True)
            stop_reason = batch["stop_reason"].to(device, non_blocking=True)
            stop_reason_mask = batch["stop_reason_mask"].to(device, non_blocking=True).reshape(-1).bool()
            out = model(scalar, layout, base_target, checkpoint_flat, speed_logits, expected_speed, camera, lidar)
            pred = out["target"]
            stop_pred = (torch.sigmoid(pred[:, -1]) >= 0.5).long()
            stop_target = (target[:, -1] >= 0.5).long()
            stop_correct += int((stop_pred == stop_target).sum().cpu())
            stop_total += int(stop_target.numel())
            state_pred = torch.argmax(out["stop_state"], dim=1)
            state_correct += int((state_pred == stop_state).sum().cpu())
            state_total += int(stop_state.numel())
            reason_pred = torch.argmax(out["stop_reason"], dim=1)
            if torch.any(stop_reason_mask):
                reason_correct += int((reason_pred[stop_reason_mask] == stop_reason[stop_reason_mask]).sum().cpu())
                reason_total += int(stop_reason_mask.sum().cpu())
            traj_dim = pred.shape[1] - 1 - int(speed_dim)
            if traj_dim > 0 and traj_dim % 3 == 0:
                pred_xy = pred[:, :traj_dim].reshape(pred.shape[0], -1, 3)[..., :2]
                target_xy = target[:, :traj_dim].reshape(target.shape[0], -1, 3)[..., :2]
                teacher_xy = teacher_target[:, :traj_dim].reshape(teacher_target.shape[0], -1, 3)[..., :2]
                xy_errors.append(torch.linalg.norm(pred_xy - target_xy, dim=-1).mean(dim=0).cpu().numpy())
                teacher_xy_errors.append(torch.linalg.norm(pred_xy - teacher_xy, dim=-1).mean(dim=0).cpu().numpy())
            pred_speed = pred[:, traj_dim : traj_dim + int(speed_dim)]
            target_speed = target[:, traj_dim : traj_dim + int(speed_dim)]
            teacher_speed = teacher_target[:, traj_dim : traj_dim + int(speed_dim)]
            base_speed = base_target[:, traj_dim : traj_dim + int(speed_dim)]
            speed_errors.append(torch.mean(torch.abs(pred_speed - target_speed), dim=1).cpu().numpy())
            teacher_speed_errors.append(torch.mean(torch.abs(pred_speed - teacher_speed), dim=1).cpu().numpy())
            if int(speed_dim) > 1:
                pred_speed_delta = pred_speed[:, 1:] - pred_speed[:, :-1]
                target_speed_delta = target_speed[:, 1:] - target_speed[:, :-1]
                speed_delta_errors.append(torch.mean(torch.abs(pred_speed_delta - target_speed_delta), dim=1).cpu().numpy())
            if int(speed_dim) > 2:
                pred_speed_curvature = pred_speed[:, 2:] - 2.0 * pred_speed[:, 1:-1] + pred_speed[:, :-2]
                target_speed_curvature = target_speed[:, 2:] - 2.0 * target_speed[:, 1:-1] + target_speed[:, :-2]
                speed_curvature_errors.append(torch.mean(torch.abs(pred_speed_curvature - target_speed_curvature), dim=1).cpu().numpy())
            pred_speeds.append(pred_speed.mean(dim=1).cpu().numpy())
            target_speeds.append(target_speed.mean(dim=1).cpu().numpy())
            teacher_speeds.append(teacher_speed.mean(dim=1).cpu().numpy())
            base_speeds.append(base_speed.mean(dim=1).cpu().numpy())
            moving = _moving_mask(scalar, target, base_target, int(speed_dim), float(moving_speed_threshold))
            moving_total += int(moving.sum().cpu())
            sample_total += int(moving.numel())
    metrics = {
        "stop_accuracy": float(stop_correct / max(stop_total, 1)),
        "stop_state_accuracy": float(state_correct / max(state_total, 1)),
        "stop_reason_accuracy": float(reason_correct / max(reason_total, 1)) if reason_total else None,
        "stop_reason_support": int(reason_total),
        "stop_state_names": STOP_STATE_NAMES,
        "stop_reason_names": STOP_REASON_NAMES,
    }
    if xy_errors:
        metrics["mean_xy_error_m_by_horizon"] = np.mean(np.stack(xy_errors, axis=0), axis=0).astype(float).tolist()
    if teacher_xy_errors:
        metrics["mean_teacher_xy_error_m_by_horizon"] = np.mean(np.stack(teacher_xy_errors, axis=0), axis=0).astype(float).tolist()
    if speed_errors:
        metrics["mean_speed_abs_error_mps"] = float(np.mean(np.concatenate(speed_errors, axis=0)))
        if teacher_speed_errors:
            metrics["mean_teacher_speed_abs_error_mps"] = float(np.mean(np.concatenate(teacher_speed_errors, axis=0)))
        if speed_delta_errors:
            metrics["mean_speed_delta_abs_error_mps"] = float(np.mean(np.concatenate(speed_delta_errors, axis=0)))
        if speed_curvature_errors:
            metrics["mean_speed_curvature_abs_error_mps"] = float(np.mean(np.concatenate(speed_curvature_errors, axis=0)))
        metrics["pred_speed_mean_mps"] = float(np.mean(np.concatenate(pred_speeds, axis=0)))
        metrics["target_speed_mean_mps"] = float(np.mean(np.concatenate(target_speeds, axis=0)))
        metrics["teacher_speed_mean_mps"] = float(np.mean(np.concatenate(teacher_speeds, axis=0)))
        metrics["base_speed_mean_mps"] = float(np.mean(np.concatenate(base_speeds, axis=0)))
        metrics["moving_ratio_eval"] = float(moving_total / max(sample_total, 1))
    return metrics


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_arrays = np.load(Path(args.cache).expanduser(), allow_pickle=True)
    metadata = _metadata_from_cache(cache_arrays)
    index = _resolve_from_metadata(args.index, metadata, "index")
    episode_root = _resolve_from_metadata(args.episode_root_override, metadata, "episode_root_override")
    if not index:
        raise ValueError("--index is required because the cache does not contain index metadata")
    if not episode_root:
        raise ValueError("--episode-root-override is required because the cache does not contain episode root metadata")
    sample_episode = cache_arrays["sample_episode"].astype(np.int64)
    train_indices, val_indices = _split_by_episode(sample_episode, val_ratio=args.val_ratio, seed=args.seed)
    if args.max_train_samples > 0:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_indices = val_indices[: args.max_val_samples]
    cameras = _camera_list(args.cameras)
    train_ds = CachedVisualPriorDataset(
        args.cache,
        index,
        episode_root,
        train_indices,
        cameras=cameras,
        image_size=tuple(args.image_size),
        lidar_size=args.lidar_size,
        use_raw_layout=args.use_raw_layout,
        teacher_cache_path=args.teacher_cache,
    )
    val_ds = CachedVisualPriorDataset(
        args.cache,
        index,
        episode_root,
        val_indices,
        cameras=cameras,
        image_size=tuple(args.image_size),
        lidar_size=args.lidar_size,
        use_raw_layout=args.use_raw_layout,
        teacher_cache_path=args.teacher_cache,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = CachedVisualTransFuserPPAdapterPolicy(
        scalar_dim=train_ds.scalar_dim,
        layout_dim=train_ds.layout_dim,
        target_dim=train_ds.target_dim,
        checkpoint_dim=int(train_ds.checkpoint_flat.shape[1]),
        speed_classes=int(train_ds.speed_logits.shape[1]),
        cameras=cameras,
        lidar_channels=train_ds.lidar_channels,
        hidden_dim=args.hidden_dim,
        layout_hidden_dim=args.layout_hidden_dim,
        visual_dim=args.visual_dim,
        visual_token_dim=args.visual_token_dim,
        visual_layers=args.visual_layers,
        visual_heads=args.visual_heads,
    )
    model.to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    print(
        json.dumps(
            {
                "mode": "transfuserpp_cached_visual_bev_adapter",
                "parameters": count_trainable_parameters(raw_model),
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "cache": str(Path(args.cache).expanduser()),
                "teacher_cache": str(Path(args.teacher_cache).expanduser()) if args.teacher_cache else "",
                "index": index,
                "episode_root_override": episode_root,
                "cameras": cameras,
                "image_size": list(args.image_size),
                "lidar_size": args.lidar_size,
                "use_raw_layout": bool(args.use_raw_layout),
                "training_strategy": {
                    "moving_speed_threshold": args.moving_speed_threshold,
                    "moving_sample_weight": args.moving_sample_weight,
                    "stopped_sample_weight": args.stopped_sample_weight,
                    "teacher_target_blend": args.teacher_target_blend,
                    "teacher_traj_blend": args.teacher_traj_blend,
                    "teacher_speed_target_blend": args.teacher_speed_target_blend,
                    "teacher_stop_target_blend": args.teacher_stop_target_blend,
                    "speed_teacher_blend": args.speed_teacher_blend,
                    "speed_distill_loss_weight": args.speed_distill_loss_weight,
                    "speed_floor_loss_weight": args.speed_floor_loss_weight,
                    "speed_floor_mps": args.speed_floor_mps,
                    "speed_delta_loss_weight": args.speed_delta_loss_weight,
                    "speed_curvature_loss_weight": args.speed_curvature_loss_weight,
                    "traj_delta_loss_weight": args.traj_delta_loss_weight,
                    "traj_curvature_loss_weight": args.traj_curvature_loss_weight,
                    "prior_loss_weight": args.prior_loss_weight,
                    "stop_loss_after_epoch": args.stop_loss_after_epoch,
                },
                "cache_metadata": metadata,
                "teacher_cache_metadata": train_ds.teacher_metadata,
            },
            indent=2,
        ),
        flush=True,
    )
    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, args, train=True, epoch=epoch)
        val_metrics = _run_epoch(model, val_loader, optimizer, device, args, train=False, epoch=epoch)
        scheduler.step()
        row = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]), "train": train_metrics, "val": val_metrics, "best_val_loss": best_val}
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            row["best_val_loss"] = best_val
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "model_state": raw_model.state_dict(),
                    "args": vars(args),
                    "scalar_dim": train_ds.scalar_dim,
                    "layout_dim": train_ds.layout_dim,
                    "target_dim": train_ds.target_dim,
                    "checkpoint_dim": int(train_ds.checkpoint_flat.shape[1]),
                    "speed_classes": int(train_ds.speed_logits.shape[1]),
                    "lidar_channels": train_ds.lidar_channels,
                    "cameras": cameras,
                    "epoch": epoch,
                    "val_loss": best_val,
                    "cache_metadata": metadata,
                    "teacher_cache_metadata": train_ds.teacher_metadata,
                },
                out_dir / "best_model.pt",
            )
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (out_dir / "latest.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"epoch={epoch:03d} train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f} best={best_val:.6f}", flush=True)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    best = torch.load(out_dir / "best_model.pt", map_location="cpu")
    raw_model.load_state_dict(best["model_state"])
    raw_model.to(device)
    metrics = _evaluate_predictions(raw_model, val_loader, device, speed_dim=args.speed_dim, moving_speed_threshold=args.moving_speed_threshold)
    metrics.update({"best_epoch": int(best["epoch"]), "best_val_loss": float(best["val_loss"]), "mode": "transfuserpp_cached_visual_bev_adapter"})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a 3-camera/LiDAR visual adapter from cached frozen TransFuser++ priors.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--teacher-cache", default="", help="Optional canonical-prior cache used as a distillation target while --cache supplies the adapter input prior.")
    parser.add_argument("--index", default="")
    parser.add_argument("--episode-root-override", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cameras", default="front,left,right")
    parser.add_argument("--image-size", type=int, nargs=2, default=[320, 180], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--lidar-size", type=int, default=128)
    parser.add_argument("--use-raw-layout", action="store_true", help="Use the layout from --index/--episode-root-override instead of the cached prior layout.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layout-hidden-dim", type=int, default=128)
    parser.add_argument("--visual-dim", type=int, default=256)
    parser.add_argument("--visual-token-dim", type=int, default=192)
    parser.add_argument("--visual-layers", type=int, default=2)
    parser.add_argument("--visual-heads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--speed-dim", type=int, default=4)
    parser.add_argument("--teacher-target-blend", type=float, default=0.0, help="Blend expert labels with teacher-cache base_target: 0=expert only, 1=canonical teacher only.")
    parser.add_argument("--teacher-traj-blend", type=float, default=None, help="Optional teacher blend for trajectory/yaw targets. Defaults to --teacher-target-blend.")
    parser.add_argument("--teacher-speed-target-blend", type=float, default=None, help="Optional teacher blend for speed targets. Defaults to --teacher-target-blend.")
    parser.add_argument("--teacher-stop-target-blend", type=float, default=None, help="Optional teacher blend for stop targets. Defaults to --teacher-target-blend.")
    parser.add_argument("--moving-speed-threshold", type=float, default=1.0)
    parser.add_argument("--moving-sample-weight", type=float, default=1.0)
    parser.add_argument("--stopped-sample-weight", type=float, default=1.0)
    parser.add_argument("--xy-loss-weight", type=float, default=1.0)
    parser.add_argument("--yaw-loss-weight", type=float, default=0.05)
    parser.add_argument("--speed-loss-weight", type=float, default=0.10)
    parser.add_argument("--speed-teacher-blend", type=float, default=0.0)
    parser.add_argument("--speed-distill-loss-weight", type=float, default=0.0)
    parser.add_argument("--speed-floor-loss-weight", type=float, default=0.0)
    parser.add_argument("--speed-floor-mps", type=float, default=1.0)
    parser.add_argument("--speed-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--speed-curvature-loss-weight", type=float, default=0.0)
    parser.add_argument("--traj-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--traj-curvature-loss-weight", type=float, default=0.0)
    parser.add_argument("--prior-loss-weight", type=float, default=0.0, help="Small regularizer toward the input prior from --cache.")
    parser.add_argument("--stop-loss-weight", type=float, default=0.05)
    parser.add_argument("--stop-state-loss-weight", type=float, default=0.10)
    parser.add_argument("--stop-reason-loss-weight", type=float, default=0.02)
    parser.add_argument("--stop-positive-loss-scale", type=float, default=1.0)
    parser.add_argument("--stop-negative-loss-scale", type=float, default=1.0)
    parser.add_argument("--stop-loss-after-epoch", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--step-log-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
