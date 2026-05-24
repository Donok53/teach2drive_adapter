import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import STOP_REASON_NAMES, STOP_STATE_NAMES
from .train_adapter import _masked_weighted_ce, _per_sample_vector, _weighted_mean
from .train_transfuserpp_fused_feature_adapter import (
    FusedFeatureCache,
    ResidualFusedFeatureAdapter,
    _feature_cosine_loss,
    _split_by_episode,
)


def _load_index_targets(index_path: str) -> Dict[str, np.ndarray]:
    arrays = np.load(Path(index_path).expanduser(), allow_pickle=True)
    traj = arrays["traj_targets"].astype(np.float32)
    speed = arrays["speed_targets"].astype(np.float32)
    stop = arrays["stop_targets"].astype(np.float32).reshape(-1, 1)
    sample_count = int(len(traj))
    return {
        "scalar": arrays["scalar_features"].astype(np.float32),
        "target": np.concatenate([traj, speed, stop], axis=1).astype(np.float32),
        "stop_state": arrays["stop_state_targets"].astype(np.int64)
        if "stop_state_targets" in arrays.files
        else np.zeros(sample_count, dtype=np.int64),
        "stop_reason": arrays["stop_reason_targets"].astype(np.int64)
        if "stop_reason_targets" in arrays.files
        else np.zeros(sample_count, dtype=np.int64),
        "stop_reason_mask": arrays["stop_reason_masks"].astype(np.float32).reshape(-1, 1)
        if "stop_reason_masks" in arrays.files
        else np.zeros((sample_count, 1), dtype=np.float32),
        "sample_weight": arrays["sample_weights"].astype(np.float32).reshape(-1, 1)
        if "sample_weights" in arrays.files
        else np.ones((sample_count, 1), dtype=np.float32),
    }


def _metadata_from_cache(arrays) -> Dict:
    if "metadata" not in arrays.files:
        return {}
    return json.loads(str(arrays["metadata"].item()))


class FusedFeaturePolicyDataset(Dataset):
    """Aligned fused features plus scalar/expert labels from the source index.

    The deployed module remains a fused-feature adapter. The auxiliary policy
    head is only used during training to make the adapted features preserve
    behavior-level information that pure feature L1 can miss.
    """

    def __init__(
        self,
        source: FusedFeatureCache,
        target: FusedFeatureCache,
        index_path: str,
        indices: Optional[np.ndarray] = None,
        teacher_cache_path: str = "",
    ) -> None:
        if source.features.shape != target.features.shape:
            raise ValueError(f"feature shape mismatch: {source.features.shape} != {target.features.shape}")
        if not np.array_equal(source.sample_episode, target.sample_episode):
            raise ValueError("source/target feature caches are not aligned by episode")
        if not np.array_equal(source.sample_frame, target.sample_frame):
            raise ValueError("source/target feature caches are not aligned by frame")
        self.source = source
        self.target = target
        self.indices = np.arange(len(source.features), dtype=np.int64) if indices is None else indices.astype(np.int64)
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
                raise ValueError("teacher cache is not aligned with source feature cache by episode/frame")
            self.teacher_target = teacher["base_target"].astype(np.float32)

    @property
    def feature_shape(self):
        return tuple(int(v) for v in self.source.features.shape[1:])

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
        return {
            "source": torch.from_numpy(np.array(self.source.features[cache_idx], copy=True)),
            "target_feature": torch.from_numpy(np.array(self.target.features[cache_idx], copy=True)),
            "scalar": torch.from_numpy(np.array(self.index_targets["scalar"][raw_idx], copy=True)),
            "target": torch.from_numpy(np.array(self.index_targets["target"][raw_idx], copy=True)),
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


class FeaturePolicyAuxiliaryAdapter(nn.Module):
    def __init__(
        self,
        channels: int,
        scalar_dim: int,
        target_dim: int,
        hidden_dim: int = 512,
        adapter_hidden_channels: int = 0,
        adapter_blocks: int = 2,
        adapter_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.adapter = ResidualFusedFeatureAdapter(
            channels=channels,
            hidden_channels=adapter_hidden_channels,
            blocks=adapter_blocks,
            dropout=adapter_dropout,
        )
        pooled_dim = int(channels) * 3 + int(scalar_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.target = nn.Linear(hidden_dim, target_dim)
        self.stop_state = nn.Linear(hidden_dim, len(STOP_STATE_NAMES))
        self.stop_reason = nn.Linear(hidden_dim, len(STOP_REASON_NAMES))

    def forward(self, source: torch.Tensor, scalar: torch.Tensor) -> Dict[str, torch.Tensor]:
        adapted = self.adapter(source)
        mean_pool = adapted.mean(dim=(2, 3))
        max_pool = adapted.amax(dim=(2, 3))
        residual_pool = (adapted - source).mean(dim=(2, 3))
        features = torch.cat([mean_pool, max_pool, residual_pool, scalar], dim=1)
        hidden = self.trunk(features)
        return {
            "adapted": adapted,
            "target": self.target(hidden),
            "stop_state": self.stop_state(hidden),
            "stop_reason": self.stop_reason(hidden),
        }


def _speed_region(pred: torch.Tensor, speed_dim: int):
    traj_dim = pred.shape[1] - int(speed_dim) - 1
    return traj_dim, pred[:, traj_dim : traj_dim + int(speed_dim)]


def _moving_mask(scalar: torch.Tensor, target: torch.Tensor, speed_dim: int, threshold: float) -> torch.Tensor:
    batch_size = int(target.shape[0])
    traj_dim, target_speed = _speed_region(target, speed_dim)
    current_speed = scalar[:, :1].abs() if scalar.shape[1] else torch.zeros_like(target_speed[:, :1])
    future_speed = target_speed.abs().amax(dim=1, keepdim=True)
    mask = ((future_speed > threshold) | (current_speed > threshold)).to(dtype=target.dtype)
    return _per_sample_vector(mask, batch_size, reduce="any").bool()


def _launch_mask(scalar: torch.Tensor, speed_target: torch.Tensor, current_threshold: float, future_threshold: float) -> torch.Tensor:
    batch_size = int(speed_target.shape[0])
    current_speed = scalar[:, :1].abs() if scalar.shape[1] else torch.zeros_like(speed_target[:, :1])
    future_speed = speed_target.clamp_min(0.0).amax(dim=1, keepdim=True)
    mask = ((current_speed <= float(current_threshold)) & (future_speed >= float(future_threshold))).to(dtype=speed_target.dtype)
    return _per_sample_vector(mask, batch_size, reduce="any").bool()


def _release_mask(stop_state: torch.Tensor, speed_target: torch.Tensor, future_threshold: float) -> torch.Tensor:
    release_id = int(STOP_STATE_NAMES.index("release_go"))
    batch_size = int(speed_target.shape[0])
    future_speed = speed_target.clamp_min(0.0).amax(dim=1, keepdim=True)
    state = stop_state.reshape(-1, 1)
    mask = ((state == release_id) & (future_speed >= float(future_threshold))).to(dtype=speed_target.dtype)
    return _per_sample_vector(mask, batch_size, reduce="any").bool()


def _target_speed_mask(speed_target: torch.Tensor, threshold: float) -> torch.Tensor:
    batch_size = int(speed_target.shape[0])
    future_speed = speed_target.clamp_min(0.0).amax(dim=1, keepdim=True)
    mask = (future_speed >= float(threshold)).to(dtype=speed_target.dtype)
    return _per_sample_vector(mask, batch_size, reduce="any").bool()


def _effective_weight(weight: torch.Tensor, moving: torch.Tensor, args) -> torch.Tensor:
    moving = moving.reshape(-1).bool()
    weight = _per_sample_vector(weight, int(moving.numel()))
    moving_scale = torch.where(
        moving,
        torch.full_like(weight, float(args.moving_sample_weight)),
        torch.full_like(weight, float(args.stopped_sample_weight)),
    )
    return weight * moving_scale


def _run_epoch(model, loader, optimizer, device, args, train: bool) -> Dict[str, float]:
    model.train(train)
    totals = {
        "loss": 0.0,
        "feature": 0.0,
        "cosine": 0.0,
        "residual": 0.0,
        "input_l1": 0.0,
        "adapted_l1": 0.0,
        "xy": 0.0,
        "yaw": 0.0,
        "speed": 0.0,
        "speed_floor": 0.0,
        "launch_floor": 0.0,
        "release_floor": 0.0,
        "stop_ceiling": 0.0,
        "stop": 0.0,
        "state": 0.0,
        "reason": 0.0,
        "moving_ratio": 0.0,
        "samples": 0,
    }
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        source = batch["source"].to(device, non_blocking=True).float()
        target_feature = batch["target_feature"].to(device, non_blocking=True).float()
        scalar = batch["scalar"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        teacher_target = batch["teacher_target"].to(device, non_blocking=True)
        stop_state = batch["stop_state"].to(device, non_blocking=True)
        stop_reason = batch["stop_reason"].to(device, non_blocking=True)
        stop_reason_mask = batch["stop_reason_mask"].to(device, non_blocking=True)
        weight = batch["sample_weight"].to(device, non_blocking=True)
        speed_dim = int(args.speed_dim)
        traj_dim = target.shape[1] - speed_dim - 1
        target_traj = (1.0 - float(args.teacher_traj_blend)) * target[:, :traj_dim] + float(args.teacher_traj_blend) * teacher_target[:, :traj_dim]
        speed_target = (
            (1.0 - float(args.teacher_speed_target_blend)) * target[:, traj_dim : traj_dim + speed_dim]
            + float(args.teacher_speed_target_blend) * teacher_target[:, traj_dim : traj_dim + speed_dim]
        )
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
        with torch.set_grad_enabled(train):
            out = model(source, scalar)
            adapted = out["adapted"]
            pred = out["target"]
            feature_loss = nn.functional.smooth_l1_loss(adapted, target_feature)
            cosine_loss = _feature_cosine_loss(adapted, target_feature)
            residual_loss = torch.mean(torch.abs(adapted - source))
            pred_traj = pred[:, :traj_dim].reshape(pred.shape[0], -1, 3)
            target_traj_view = target_traj.reshape(target_traj.shape[0], -1, 3)
            xy_loss = _weighted_mean(torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., :2], target_traj_view[..., :2], reduction="none"), dim=(1, 2)), effective_weight)
            yaw_loss = _weighted_mean(torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., 2], target_traj_view[..., 2], reduction="none"), dim=1), effective_weight)
            pred_speed = pred[:, traj_dim : traj_dim + speed_dim]
            speed_loss = _weighted_mean(
                torch.mean(nn.functional.smooth_l1_loss(pred_speed, speed_target, reduction="none"), dim=1),
                effective_weight,
            )
            target_go = _target_speed_mask(speed_target, float(args.speed_floor_target_threshold))
            target_stop = ~_target_speed_mask(speed_target, float(args.stop_speed_target_threshold))
            speed_floor_raw = torch.relu(float(args.speed_floor_mps) - pred_speed).mean(dim=1) * target_go.to(dtype=pred_speed.dtype)
            speed_floor_loss = _weighted_mean(speed_floor_raw, effective_weight)
            launch_floor_raw = torch.relu(float(args.launch_speed_floor_mps) - pred_speed).mean(dim=1) * launch.to(dtype=pred_speed.dtype)
            launch_floor_loss = _weighted_mean(launch_floor_raw, effective_weight)
            release_floor_raw = torch.relu(float(args.release_speed_floor_mps) - pred_speed).mean(dim=1) * release.to(dtype=pred_speed.dtype)
            release_floor_loss = _weighted_mean(release_floor_raw, effective_weight)
            stop_ceiling_raw = torch.relu(pred_speed - float(args.stop_speed_ceiling_mps)).mean(dim=1) * target_stop.to(dtype=pred_speed.dtype)
            stop_ceiling_loss = _weighted_mean(stop_ceiling_raw, effective_weight)
            stop_loss = _weighted_mean(nn.functional.binary_cross_entropy_with_logits(pred[:, -1:], stop_target, reduction="none"), effective_weight)
            state_loss = _weighted_mean(nn.functional.cross_entropy(out["stop_state"], stop_state, reduction="none"), effective_weight)
            reason_loss = _masked_weighted_ce(out["stop_reason"], stop_reason, stop_reason_mask, effective_weight)
            loss = (
                float(args.feature_loss_weight) * feature_loss
                + float(args.cosine_loss_weight) * cosine_loss
                + float(args.residual_loss_weight) * residual_loss
                + float(args.xy_loss_weight) * xy_loss
                + float(args.yaw_loss_weight) * yaw_loss
                + float(args.speed_loss_weight) * speed_loss
                + float(args.speed_floor_loss_weight) * speed_floor_loss
                + float(args.launch_speed_floor_loss_weight) * launch_floor_loss
                + float(args.release_speed_floor_loss_weight) * release_floor_loss
                + float(args.stop_speed_ceiling_loss_weight) * stop_ceiling_loss
                + float(args.stop_loss_weight) * stop_loss
                + float(args.stop_state_loss_weight) * state_loss
                + float(args.stop_reason_loss_weight) * reason_loss
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(args.grad_clip) > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()
        batch_size = int(source.shape[0])
        input_l1 = torch.mean(torch.abs(source - target_feature))
        adapted_l1 = torch.mean(torch.abs(adapted.detach() - target_feature))
        for key, value in (
            ("loss", loss),
            ("feature", feature_loss),
            ("cosine", cosine_loss),
            ("residual", residual_loss),
            ("input_l1", input_l1),
            ("adapted_l1", adapted_l1),
            ("xy", xy_loss),
            ("yaw", yaw_loss),
            ("speed", speed_loss),
            ("speed_floor", speed_floor_loss),
            ("launch_floor", launch_floor_loss),
            ("release_floor", release_floor_loss),
            ("stop_ceiling", stop_ceiling_loss),
            ("stop", stop_loss),
            ("state", state_loss),
            ("reason", reason_loss),
        ):
            totals[key] += float(value.detach().cpu()) * batch_size
        totals["moving_ratio"] += float(moving.float().mean().detach().cpu()) * batch_size
        totals["samples"] += batch_size
        if train and int(args.step_log_every) > 0 and (step == 1 or step % int(args.step_log_every) == 0):
            samples = max(totals["samples"], 1)
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"step={step:05d}/{len(loader):05d} loss={totals['loss']/samples:.6f} "
                f"feature={totals['feature']/samples:.6f} adapted_l1={totals['adapted_l1']/samples:.6f} "
                f"xy={totals['xy']/samples:.6f} speed={totals['speed']/samples:.6f} "
                f"floor={totals['speed_floor']/samples:.6f} launch={totals['launch_floor']/samples:.6f} "
                f"stopceil={totals['stop_ceiling']/samples:.6f} moving={totals['moving_ratio']/samples:.3f} "
                f"samples/s={samples/elapsed:.1f}",
                flush=True,
            )
    samples = max(int(totals.pop("samples")), 1)
    return {key: value / samples for key, value in totals.items()}


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_cache = FusedFeatureCache(args.source_cache)
    target_cache = FusedFeatureCache(args.target_cache)
    index_path = args.index or str(source_cache.metadata.get("index", ""))
    if not index_path:
        raise ValueError("--index is required because source feature cache metadata has no index")
    train_indices, val_indices = _split_by_episode(source_cache.sample_episode, float(args.val_ratio), int(args.seed))
    if int(args.max_train_samples) > 0:
        train_indices = train_indices[: int(args.max_train_samples)]
    if int(args.max_val_samples) > 0:
        val_indices = val_indices[: int(args.max_val_samples)]
    train_ds = FusedFeaturePolicyDataset(source_cache, target_cache, index_path, train_indices, teacher_cache_path=args.teacher_cache)
    val_ds = FusedFeaturePolicyDataset(source_cache, target_cache, index_path, val_indices, teacher_cache_path=args.teacher_cache)
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(), drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=torch.cuda.is_available(), drop_last=False)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    feature_shape = train_ds.feature_shape
    if len(feature_shape) != 3:
        raise ValueError(f"Expected feature shape [C,H,W], got {feature_shape}")
    model = FeaturePolicyAuxiliaryAdapter(
        channels=feature_shape[0],
        scalar_dim=train_ds.scalar_dim,
        target_dim=train_ds.target_dim,
        hidden_dim=int(args.hidden_dim),
        adapter_hidden_channels=int(args.hidden_channels),
        adapter_blocks=int(args.blocks),
        adapter_dropout=float(args.dropout),
    ).to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    metadata = {
        "mode": "transfuserpp_fused_feature_policy_adapter",
        "source_cache": str(Path(args.source_cache).expanduser()),
        "target_cache": str(Path(args.target_cache).expanduser()),
        "teacher_cache": str(Path(args.teacher_cache).expanduser()) if args.teacher_cache else "",
        "index": str(Path(index_path).expanduser()),
        "source_metadata": source_cache.metadata,
        "target_metadata": target_cache.metadata,
        "teacher_metadata": train_ds.teacher_metadata,
        "feature_shape": list(feature_shape),
        "scalar_dim": int(train_ds.scalar_dim),
        "target_dim": int(train_ds.target_dim),
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "args": vars(args),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"train_samples": len(train_ds), "val_samples": len(val_ds), "feature_shape": feature_shape}, indent=2), flush=True)
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_metrics = _run_epoch(model, val_loader, optimizer, device, args, train=False)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        val_loss = float(val_metrics["loss"])
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.6f} val={val_loss:.6f} "
            f"best={best_val if best_epoch else val_loss:.6f} input_l1={val_metrics['input_l1']:.6f} "
            f"adapted_l1={val_metrics['adapted_l1']:.6f}",
            flush=True,
        )
        if val_loss + float(args.early_stop_min_delta) < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "model_state": raw_model.adapter.state_dict(),
                    "aux_model_state": raw_model.state_dict(),
                    "feature_shape": feature_shape,
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
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {"best_epoch": int(best_epoch), "best_val_loss": float(best_val), "mode": "transfuserpp_fused_feature_policy_adapter"}
    if (out_dir / "best_model.pt").exists():
        best = torch.load(out_dir / "best_model.pt", map_location="cpu")
        summary.update(best.get("val_metrics", {}))
    print(json.dumps(summary, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a fused-feature adapter with auxiliary policy supervision.")
    parser.add_argument("--source-cache", required=True, help="Shifted feature cache directory.")
    parser.add_argument("--target-cache", required=True, help="Canonical feature cache directory.")
    parser.add_argument("--teacher-cache", default="", help="Optional canonical prior cache for teacher target blending.")
    parser.add_argument("--index", default="", help="Source Teach2Drive index. Defaults to source cache metadata.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--hidden-channels", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--feature-loss-weight", type=float, default=1.0)
    parser.add_argument("--cosine-loss-weight", type=float, default=0.05)
    parser.add_argument("--residual-loss-weight", type=float, default=0.01)
    parser.add_argument("--xy-loss-weight", type=float, default=0.35)
    parser.add_argument("--yaw-loss-weight", type=float, default=0.02)
    parser.add_argument("--speed-loss-weight", type=float, default=0.15)
    parser.add_argument("--stop-loss-weight", type=float, default=0.02)
    parser.add_argument("--stop-state-loss-weight", type=float, default=0.02)
    parser.add_argument("--stop-reason-loss-weight", type=float, default=0.01)
    parser.add_argument("--teacher-traj-blend", type=float, default=0.0)
    parser.add_argument("--teacher-speed-target-blend", type=float, default=0.0)
    parser.add_argument("--teacher-stop-target-blend", type=float, default=0.0)
    parser.add_argument("--speed-dim", type=int, default=4)
    parser.add_argument("--moving-speed-threshold", type=float, default=1.0)
    parser.add_argument("--moving-sample-weight", type=float, default=1.0)
    parser.add_argument("--stopped-sample-weight", type=float, default=1.0)
    parser.add_argument("--speed-floor-loss-weight", type=float, default=0.05)
    parser.add_argument("--speed-floor-mps", type=float, default=1.0)
    parser.add_argument("--speed-floor-target-threshold", type=float, default=2.0)
    parser.add_argument("--stop-speed-ceiling-loss-weight", type=float, default=0.05)
    parser.add_argument("--stop-speed-ceiling-mps", type=float, default=0.5)
    parser.add_argument("--stop-speed-target-threshold", type=float, default=0.5)
    parser.add_argument("--launch-current-speed-threshold", type=float, default=0.8)
    parser.add_argument("--launch-target-speed-threshold", type=float, default=2.0)
    parser.add_argument("--launch-sample-weight", type=float, default=1.5)
    parser.add_argument("--launch-speed-floor-loss-weight", type=float, default=0.05)
    parser.add_argument("--launch-speed-floor-mps", type=float, default=2.0)
    parser.add_argument("--release-target-speed-threshold", type=float, default=1.0)
    parser.add_argument("--release-sample-weight", type=float, default=1.5)
    parser.add_argument("--release-speed-floor-loss-weight", type=float, default=0.05)
    parser.add_argument("--release-speed-floor-mps", type=float, default=2.0)
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
