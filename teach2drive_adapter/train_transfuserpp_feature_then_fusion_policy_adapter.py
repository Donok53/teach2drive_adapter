from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .cache_transfuserpp_feature_fusion_features import FUSED_FEATURE_NAME
from .data import STOP_REASON_NAMES, STOP_STATE_NAMES
from .train_adapter import _masked_weighted_ce, _per_sample_vector, _weighted_mean
from .train_transfuserpp_feature_then_fusion_adapter import (
    ExtrinsicAwareFeatureThenFusionAdapter,
    FeatureFusionCache,
    FeatureThenFusionAdapter,
    _map_losses,
    _move_feature_dict,
    _shape_dict,
    build_extrinsic_vector,
    load_feature_then_fusion_checkpoint,
)
from .train_transfuserpp_fused_feature_adapter import _feature_cosine_loss, _split_by_episode
from .train_transfuserpp_fused_feature_policy_adapter import (
    _effective_weight,
    _launch_mask,
    _load_index_targets,
    _metadata_from_cache,
    _moving_mask,
    _release_mask,
    _target_speed_mask,
)


class FeatureFusionPolicyPairDataset(Dataset):
    """Aligned feature+fusion caches with source-index policy labels.

    The deployed checkpoint is still a normal feature-then-fusion adapter.  The
    policy head below exists only during training so behavior-critical gradients
    reach the fused feature adapter.
    """

    def __init__(
        self,
        source: FeatureFusionCache,
        target: FeatureFusionCache,
        index_path: str,
        indices: Optional[np.ndarray] = None,
        teacher_cache_path: str = "",
    ) -> None:
        if source.feature_names != target.feature_names:
            raise ValueError(f"feature name mismatch: {source.feature_names} != {target.feature_names}")
        for name in source.feature_names:
            if source.features[name].shape != target.features[name].shape:
                raise ValueError(f"{name} shape mismatch: {source.features[name].shape} != {target.features[name].shape}")
        if not np.array_equal(source.sample_episode, target.sample_episode):
            raise ValueError("source/target caches are not aligned by episode")
        if not np.array_equal(source.sample_frame, target.sample_frame):
            raise ValueError("source/target caches are not aligned by frame")

        self.source = source
        self.target = target
        self.indices = np.arange(len(source.sample_episode), dtype=np.int64) if indices is None else indices.astype(np.int64)
        self.index_path = str(Path(index_path).expanduser())
        self.index_targets = _load_index_targets(self.index_path)
        self.teacher_cache_path = str(Path(teacher_cache_path).expanduser()) if teacher_cache_path else ""
        self.teacher_metadata: Dict = {}
        self.teacher_target = None
        if self.teacher_cache_path:
            teacher = np.load(self.teacher_cache_path, allow_pickle=True)
            self.teacher_metadata = _metadata_from_cache(teacher)
            teacher_episode = teacher["sample_episode"].astype(np.int64)
            teacher_frame = teacher["sample_frame"].astype(np.int64)
            if not np.array_equal(teacher_episode, source.sample_episode) or not np.array_equal(teacher_frame, source.sample_frame):
                raise ValueError("teacher cache is not aligned with feature-fusion cache by episode/frame")
            self.teacher_target = teacher["base_target"].astype(np.float32)

    @property
    def scalar_dim(self) -> int:
        return int(self.index_targets["scalar"].shape[1])

    @property
    def target_dim(self) -> int:
        return int(self.index_targets["target"].shape[1])

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        cache_idx = int(self.indices[item])
        raw_idx = int(self.source.sample_index[cache_idx])
        source = {
            name: torch.from_numpy(np.array(self.source.features[name][cache_idx], copy=True))
            for name in self.source.feature_names
        }
        target = {
            name: torch.from_numpy(np.array(self.target.features[name][cache_idx], copy=True))
            for name in self.target.feature_names
        }
        return {
            "source": source,
            "target": target,
            "scalar": torch.from_numpy(np.array(self.index_targets["scalar"][raw_idx], copy=True)),
            "policy_target": torch.from_numpy(np.array(self.index_targets["target"][raw_idx], copy=True)),
            "teacher_target": torch.from_numpy(
                np.array(self.teacher_target[cache_idx], copy=True)
                if self.teacher_target is not None
                else np.array(self.index_targets["target"][raw_idx], copy=True)
            ),
            "stop_state": torch.tensor(int(self.index_targets["stop_state"][raw_idx]), dtype=torch.long),
            "stop_reason": torch.tensor(int(self.index_targets["stop_reason"][raw_idx]), dtype=torch.long),
            "stop_reason_mask": torch.from_numpy(np.array(self.index_targets["stop_reason_mask"][raw_idx], copy=True)),
            "sample_weight": torch.from_numpy(np.array(self.index_targets["sample_weight"][raw_idx], copy=True)),
            "episode": torch.tensor(int(self.source.sample_episode[cache_idx]), dtype=torch.long),
            "frame": torch.tensor(int(self.source.sample_frame[cache_idx]), dtype=torch.long),
        }


class FusedPolicyAuxHead(nn.Module):
    def __init__(self, channels: int, scalar_dim: int, target_dim: int, hidden_dim: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        pooled_dim = int(channels) * 3 + int(scalar_dim)
        layers: list[nn.Module] = [
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, int(hidden_dim)),
            nn.GELU(),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        layers.extend(
            [
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.GELU(),
            ]
        )
        self.trunk = nn.Sequential(*layers)
        self.target = nn.Linear(int(hidden_dim), int(target_dim))
        self.stop_state = nn.Linear(int(hidden_dim), len(STOP_STATE_NAMES))
        self.stop_reason = nn.Linear(int(hidden_dim), len(STOP_REASON_NAMES))

    def forward(self, adapted_fused: torch.Tensor, source_fused: torch.Tensor, scalar: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean_pool = adapted_fused.mean(dim=(2, 3))
        max_pool = adapted_fused.amax(dim=(2, 3))
        residual_pool = (adapted_fused - source_fused).mean(dim=(2, 3))
        hidden = self.trunk(torch.cat([mean_pool, max_pool, residual_pool, scalar], dim=1))
        return {
            "target": self.target(hidden),
            "stop_state": self.stop_state(hidden),
            "stop_reason": self.stop_reason(hidden),
        }


def _policy_losses(out: Dict[str, torch.Tensor], batch: Dict, device: torch.device, args) -> Dict[str, torch.Tensor]:
    scalar = batch["scalar"].to(device, non_blocking=True)
    target = batch["policy_target"].to(device, non_blocking=True)
    teacher_target = batch["teacher_target"].to(device, non_blocking=True)
    stop_state = batch["stop_state"].to(device, non_blocking=True)
    stop_reason = batch["stop_reason"].to(device, non_blocking=True)
    stop_reason_mask = batch["stop_reason_mask"].to(device, non_blocking=True)
    weight = batch["sample_weight"].to(device, non_blocking=True)

    speed_dim = int(args.speed_dim)
    traj_dim = target.shape[1] - speed_dim - 1
    target_traj = (
        (1.0 - float(args.teacher_traj_blend)) * target[:, :traj_dim]
        + float(args.teacher_traj_blend) * teacher_target[:, :traj_dim]
    )
    speed_target = (
        (1.0 - float(args.teacher_speed_target_blend)) * target[:, traj_dim : traj_dim + speed_dim]
        + float(args.teacher_speed_target_blend) * teacher_target[:, traj_dim : traj_dim + speed_dim]
    )
    # The cached teacher stores a TF++ stop proxy/logit in the last target slot,
    # not a BCE probability. Convert it before blending so BCE cannot become
    # negative and drive the feature adapter to explode.
    teacher_stop_target = torch.sigmoid(teacher_target[:, -1:])
    stop_target = (
        (1.0 - float(args.teacher_stop_target_blend)) * target[:, -1:]
        + float(args.teacher_stop_target_blend) * teacher_stop_target
    ).clamp(0.0, 1.0)
    supervision_target = torch.cat([target_traj, speed_target, stop_target], dim=1)

    moving = _moving_mask(scalar, supervision_target, speed_dim, float(args.moving_speed_threshold))
    launch = _launch_mask(scalar, speed_target, float(args.launch_current_speed_threshold), float(args.launch_target_speed_threshold))
    release = _release_mask(stop_state, speed_target, float(args.release_target_speed_threshold))
    effective_weight = _effective_weight(weight, moving, args)
    if float(args.launch_sample_weight) != 1.0:
        effective_weight = effective_weight * torch.where(
            launch,
            torch.full_like(effective_weight, float(args.launch_sample_weight)),
            torch.ones_like(effective_weight),
        )
    if float(args.release_sample_weight) != 1.0:
        effective_weight = effective_weight * torch.where(
            release,
            torch.full_like(effective_weight, float(args.release_sample_weight)),
            torch.ones_like(effective_weight),
        )

    pred = out["target"]
    pred_traj = pred[:, :traj_dim].reshape(pred.shape[0], -1, 3)
    target_traj_view = target_traj.reshape(target_traj.shape[0], -1, 3)
    xy_loss = _weighted_mean(
        torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., :2], target_traj_view[..., :2], reduction="none"), dim=(1, 2)),
        effective_weight,
    )
    yaw_loss = _weighted_mean(
        torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., 2], target_traj_view[..., 2], reduction="none"), dim=1),
        effective_weight,
    )
    pred_speed = pred[:, traj_dim : traj_dim + speed_dim]
    speed_loss = _weighted_mean(
        torch.mean(nn.functional.smooth_l1_loss(pred_speed, speed_target, reduction="none"), dim=1),
        effective_weight,
    )

    target_go = _target_speed_mask(speed_target, float(args.speed_floor_target_threshold))
    target_stop = ~_target_speed_mask(speed_target, float(args.stop_speed_target_threshold))
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
    stop_loss = _weighted_mean(nn.functional.binary_cross_entropy_with_logits(pred[:, -1:], stop_target, reduction="none"), effective_weight)
    state_loss = _weighted_mean(nn.functional.cross_entropy(out["stop_state"], stop_state, reduction="none"), effective_weight)
    reason_loss = _masked_weighted_ce(out["stop_reason"], stop_reason, stop_reason_mask, effective_weight)
    total = (
        float(args.xy_loss_weight) * xy_loss
        + float(args.yaw_loss_weight) * yaw_loss
        + float(args.speed_loss_weight) * speed_loss
        + float(args.speed_floor_loss_weight) * speed_floor
        + float(args.launch_speed_floor_loss_weight) * launch_floor
        + float(args.release_speed_floor_loss_weight) * release_floor
        + float(args.stop_speed_ceiling_loss_weight) * stop_ceiling
        + float(args.stop_loss_weight) * stop_loss
        + float(args.stop_state_loss_weight) * state_loss
        + float(args.stop_reason_loss_weight) * reason_loss
    )
    return {
        "policy": total,
        "xy": xy_loss,
        "yaw": yaw_loss,
        "speed": speed_loss,
        "speed_floor": speed_floor,
        "launch_floor": launch_floor,
        "release_floor": release_floor,
        "stop_ceiling": stop_ceiling,
        "stop": stop_loss,
        "state": state_loss,
        "reason": reason_loss,
        "moving_ratio": moving.float().mean(),
        "launch_ratio": launch.float().mean(),
        "release_ratio": release.float().mean(),
    }


def _run_epoch(model, aux_head, loader, optimizer, device, args, train: bool, epoch: int) -> Dict[str, float]:
    model.train(train)
    aux_head.train(train)
    metric_names = (
        "loss",
        "feature_loss",
        "policy",
        "stage_feature",
        "stage_adapted_l1",
        "fused_feature",
        "fused_adapted_l1",
        "xy",
        "yaw",
        "speed",
        "speed_floor",
        "launch_floor",
        "release_floor",
        "stop_ceiling",
        "stop",
        "state",
        "reason",
        "moving_ratio",
        "launch_ratio",
        "release_ratio",
    )
    totals = {name: 0.0 for name in metric_names}
    totals["samples"] = 0
    start = time.time()
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    stage_names = tuple(raw_model.stage_names)
    policy_active = int(epoch) >= int(args.policy_loss_after_epoch)

    for step, batch in enumerate(loader, start=1):
        source = _move_feature_dict(batch["source"], device)
        target = _move_feature_dict(batch["target"], device)
        scalar = batch["scalar"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            pred = model(source)
            base_reference = None
            if (
                isinstance(raw_model, ExtrinsicAwareFeatureThenFusionAdapter)
                and (
                    float(args.stage_base_consistency_loss_weight) > 0.0
                    or float(args.fused_base_consistency_loss_weight) > 0.0
                )
            ):
                with torch.no_grad():
                    base_reference = {name: value.detach() for name, value in raw_model.base(source).items()}

            stage_losses = []
            stage_feature = []
            stage_adapted_l1 = []
            for name in stage_names:
                values = _map_losses(
                    pred[name],
                    source[name],
                    target[name],
                    float(args.stage_feature_loss_weight),
                    float(args.stage_cosine_loss_weight),
                    float(args.stage_residual_loss_weight),
                    base_reference[name] if base_reference is not None else None,
                    float(args.stage_base_consistency_loss_weight),
                )
                stage_losses.append(values[0])
                stage_feature.append(values[1])
                stage_adapted_l1.append(values[6])
            stage_loss = torch.stack(stage_losses).mean()
            fused_values = _map_losses(
                pred[FUSED_FEATURE_NAME],
                source[FUSED_FEATURE_NAME],
                target[FUSED_FEATURE_NAME],
                float(args.fused_feature_loss_weight),
                float(args.fused_cosine_loss_weight),
                float(args.fused_residual_loss_weight),
                base_reference[FUSED_FEATURE_NAME] if base_reference is not None else None,
                float(args.fused_base_consistency_loss_weight),
            )
            feature_loss = float(args.stage_loss_weight) * stage_loss + float(args.fused_loss_weight) * fused_values[0]
            aux_out = aux_head(pred[FUSED_FEATURE_NAME], source[FUSED_FEATURE_NAME], scalar)
            policy_losses = _policy_losses(aux_out, batch, device, args)
            policy_loss = policy_losses["policy"] if policy_active else feature_loss.new_tensor(0.0)
            loss = feature_loss + float(args.policy_loss_weight) * policy_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(args.grad_clip) > 0.0:
                    nn.utils.clip_grad_norm_(
                        [param for group in optimizer.param_groups for param in group["params"]],
                        float(args.grad_clip),
                    )
                optimizer.step()

        batch_size = int(source[FUSED_FEATURE_NAME].shape[0])
        values = {
            "loss": loss,
            "feature_loss": feature_loss,
            "policy": policy_loss,
            "stage_feature": torch.stack(stage_feature).mean(),
            "stage_adapted_l1": torch.stack(stage_adapted_l1).mean(),
            "fused_feature": fused_values[1],
            "fused_adapted_l1": fused_values[6],
            **policy_losses,
        }
        values["policy"] = policy_loss
        for name, value in values.items():
            totals[name] += float(value.detach().cpu()) * batch_size
        totals["samples"] += batch_size
        if train and int(args.step_log_every) > 0 and (step == 1 or step % int(args.step_log_every) == 0):
            samples = max(totals["samples"], 1)
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"step={step:05d}/{len(loader):05d} loss={totals['loss']/samples:.6f} "
                f"feat={totals['feature_loss']/samples:.6f} policy={totals['policy']/samples:.6f} "
                f"stage_l1={totals['stage_adapted_l1']/samples:.6f} fused_l1={totals['fused_adapted_l1']/samples:.6f} "
                f"xy={totals['xy']/samples:.6f} speed={totals['speed']/samples:.6f} "
                f"floor={totals['speed_floor']/samples:.6f} launch={totals['launch_floor']/samples:.6f} "
                f"moving={totals['moving_ratio']/samples:.3f} samples/s={samples/elapsed:.1f}",
                flush=True,
            )
    samples = max(int(totals.pop("samples")), 1)
    return {key: value / samples for key, value in totals.items()}


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_cache = FeatureFusionCache(args.source_cache)
    target_cache = FeatureFusionCache(args.target_cache)
    index_path = args.index or str(source_cache.metadata.get("index", ""))
    if not index_path:
        raise ValueError("--index is required because source cache metadata has no index")
    train_indices, val_indices = _split_by_episode(source_cache.sample_episode, float(args.val_ratio), int(args.seed))
    if int(args.max_train_samples) > 0:
        train_indices = train_indices[: int(args.max_train_samples)]
    if int(args.max_val_samples) > 0:
        val_indices = val_indices[: int(args.max_val_samples)]

    train_ds = FeatureFusionPolicyPairDataset(source_cache, target_cache, index_path, train_indices, args.teacher_cache)
    val_ds = FeatureFusionPolicyPairDataset(source_cache, target_cache, index_path, val_indices, args.teacher_cache)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
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
    stage_feature_shapes = _shape_dict(source_cache)
    fused_feature_shape = tuple(int(v) for v in source_cache.features[source_cache.fused_name].shape[1:])
    extrinsic_vector = build_extrinsic_vector(args.source_profile)
    if args.extrinsic_aware:
        model = ExtrinsicAwareFeatureThenFusionAdapter(
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
        model = FeatureThenFusionAdapter(
            stage_feature_shapes=stage_feature_shapes,
            fused_feature_shape=fused_feature_shape,
            hidden_channels=int(args.hidden_channels),
            blocks=int(args.blocks),
            dropout=float(args.dropout),
        ).to(device)
    load_info = load_feature_then_fusion_checkpoint(model, args.init_checkpoint, strict=False)
    if args.freeze_base:
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        if isinstance(raw_model, ExtrinsicAwareFeatureThenFusionAdapter):
            for param in raw_model.base.parameters():
                param.requires_grad_(False)

    aux_head = FusedPolicyAuxHead(
        channels=int(fused_feature_shape[0]),
        scalar_dim=train_ds.scalar_dim,
        target_dim=train_ds.target_dim,
        hidden_dim=int(args.policy_hidden_dim),
        dropout=float(args.policy_dropout),
    ).to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)

    trainable = [param for param in model.parameters() if param.requires_grad]
    trainable.extend(param for param in aux_head.parameters() if param.requires_grad)
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))
    metadata = {
        "mode": "transfuserpp_extrinsic_feature_then_fusion_policy_adapter" if args.extrinsic_aware else "transfuserpp_feature_then_fusion_policy_adapter",
        "source_cache": str(Path(args.source_cache).expanduser()),
        "target_cache": str(Path(args.target_cache).expanduser()),
        "teacher_cache": str(Path(args.teacher_cache).expanduser()) if args.teacher_cache else "",
        "index": str(Path(index_path).expanduser()),
        "source_metadata": source_cache.metadata,
        "target_metadata": target_cache.metadata,
        "teacher_metadata": train_ds.teacher_metadata,
        "stage_feature_shapes": {key: list(value) for key, value in stage_feature_shapes.items()},
        "fused_feature_shape": list(fused_feature_shape),
        "extrinsic_aware": bool(args.extrinsic_aware),
        "source_profile": args.source_profile,
        "extrinsic_vector": list(float(v) for v in extrinsic_vector),
        "init_checkpoint": str(Path(args.init_checkpoint).expanduser()) if args.init_checkpoint else "",
        "init_load_info": load_info,
        "scalar_dim": int(train_ds.scalar_dim),
        "target_dim": int(train_ds.target_dim),
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "args": vars(args),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "stage_feature_shapes": metadata["stage_feature_shapes"],
                "fused_feature_shape": list(fused_feature_shape),
                "extrinsic_aware": bool(args.extrinsic_aware),
                "source_profile": args.source_profile,
                "init_load_info": load_info,
                "teacher_cache": metadata["teacher_cache"],
            },
            indent=2,
        ),
        flush=True,
    )

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(model, aux_head, train_loader, optimizer, device, args, train=True, epoch=epoch)
        val_metrics = _run_epoch(model, aux_head, val_loader, optimizer, device, args, train=False, epoch=epoch)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        val_loss = float(val_metrics["loss"])
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.6f} val={val_loss:.6f} "
            f"best={best_val if best_epoch else val_loss:.6f} "
            f"feature={val_metrics['feature_loss']:.6f} policy={val_metrics['policy']:.6f} "
            f"stage_l1={val_metrics['stage_adapted_l1']:.6f} fused_l1={val_metrics['fused_adapted_l1']:.6f}",
            flush=True,
        )
        if val_loss + float(args.early_stop_min_delta) < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": raw_model.state_dict(),
                    "aux_model_state": aux_head.state_dict(),
                    "stage_feature_shapes": stage_feature_shapes,
                    "fused_feature_shape": fused_feature_shape,
                    "metadata": metadata,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                },
                out_dir / "best_model.pt",
            )
        else:
            stale += 1
            if stale >= int(args.early_stop_patience):
                print(f"early_stop: no val improvement for {stale} epochs (patience={args.early_stop_patience}, best={best_val:.6f})", flush=True)
                break
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {"best_epoch": int(best_epoch), "best_val_loss": float(best_val), "mode": metadata["mode"]}
    if history and not (out_dir / "best_model.pt").exists():
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        last = history[-1]
        torch.save(
            {
                "model_state": raw_model.state_dict(),
                "aux_model_state": aux_head.state_dict(),
                "stage_feature_shapes": stage_feature_shapes,
                "fused_feature_shape": fused_feature_shape,
                "metadata": metadata,
                "epoch": int(last["epoch"]),
                "val_metrics": last["val"],
                "train_metrics": last["train"],
                "recovered_missing_best": True,
            },
            out_dir / "best_model.pt",
        )
        print("warning: best_model.pt was missing at shutdown; saved final epoch checkpoint instead", flush=True)
    if (out_dir / "best_model.pt").exists():
        best = torch.load(out_dir / "best_model.pt", map_location="cpu")
        summary.update(best.get("val_metrics", {}))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a feature-then-fusion adapter with auxiliary canonical-policy recovery.")
    parser.add_argument("--source-cache", required=True, help="Shifted feature-fusion cache directory.")
    parser.add_argument("--target-cache", required=True, help="Canonical feature-fusion cache directory.")
    parser.add_argument("--teacher-cache", default="", help="Optional canonical TransFuser++ prior cache for policy target blending.")
    parser.add_argument("--index", default="", help="Source Teach2Drive index. Defaults to source cache metadata.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--extrinsic-aware", action="store_true")
    parser.add_argument("--source-profile", default="front_triplet_shifted", choices=["front_triplet_shifted", "tfpp_ego"])
    parser.add_argument("--extrinsic-hidden-dim", type=int, default=64)
    parser.add_argument("--extrinsic-dropout", type=float, default=0.0)
    parser.add_argument("--freeze-base", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--hidden-channels", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--stage-loss-weight", type=float, default=1.0)
    parser.add_argument("--fused-loss-weight", type=float, default=1.0)
    parser.add_argument("--stage-feature-loss-weight", type=float, default=1.0)
    parser.add_argument("--stage-cosine-loss-weight", type=float, default=0.04)
    parser.add_argument("--stage-residual-loss-weight", type=float, default=0.035)
    parser.add_argument("--stage-base-consistency-loss-weight", type=float, default=0.10)
    parser.add_argument("--fused-feature-loss-weight", type=float, default=1.0)
    parser.add_argument("--fused-cosine-loss-weight", type=float, default=0.04)
    parser.add_argument("--fused-residual-loss-weight", type=float, default=0.045)
    parser.add_argument("--fused-base-consistency-loss-weight", type=float, default=0.18)
    parser.add_argument("--policy-loss-weight", type=float, default=0.45)
    parser.add_argument("--policy-loss-after-epoch", type=int, default=1)
    parser.add_argument("--policy-hidden-dim", type=int, default=512)
    parser.add_argument("--policy-dropout", type=float, default=0.02)
    parser.add_argument("--xy-loss-weight", type=float, default=0.40)
    parser.add_argument("--yaw-loss-weight", type=float, default=0.03)
    parser.add_argument("--speed-loss-weight", type=float, default=0.22)
    parser.add_argument("--stop-loss-weight", type=float, default=0.04)
    parser.add_argument("--stop-state-loss-weight", type=float, default=0.04)
    parser.add_argument("--stop-reason-loss-weight", type=float, default=0.02)
    parser.add_argument("--teacher-traj-blend", type=float, default=0.75)
    parser.add_argument("--teacher-speed-target-blend", type=float, default=0.45)
    parser.add_argument("--teacher-stop-target-blend", type=float, default=0.0)
    parser.add_argument("--speed-dim", type=int, default=4)
    parser.add_argument("--moving-speed-threshold", type=float, default=1.0)
    parser.add_argument("--moving-sample-weight", type=float, default=1.35)
    parser.add_argument("--stopped-sample-weight", type=float, default=1.0)
    parser.add_argument("--speed-floor-loss-weight", type=float, default=0.04)
    parser.add_argument("--speed-floor-mps", type=float, default=1.0)
    parser.add_argument("--speed-floor-target-threshold", type=float, default=2.0)
    parser.add_argument("--stop-speed-ceiling-loss-weight", type=float, default=0.04)
    parser.add_argument("--stop-speed-ceiling-mps", type=float, default=0.5)
    parser.add_argument("--stop-speed-target-threshold", type=float, default=0.5)
    parser.add_argument("--launch-current-speed-threshold", type=float, default=0.8)
    parser.add_argument("--launch-target-speed-threshold", type=float, default=2.0)
    parser.add_argument("--launch-sample-weight", type=float, default=1.6)
    parser.add_argument("--launch-speed-floor-loss-weight", type=float, default=0.08)
    parser.add_argument("--launch-speed-floor-mps", type=float, default=2.0)
    parser.add_argument("--release-target-speed-threshold", type=float, default=1.0)
    parser.add_argument("--release-sample-weight", type=float, default=1.5)
    parser.add_argument("--release-speed-floor-loss-weight", type=float, default=0.06)
    parser.add_argument("--release-speed-floor-mps", type=float, default=1.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--step-log-every", type=int, default=50)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
