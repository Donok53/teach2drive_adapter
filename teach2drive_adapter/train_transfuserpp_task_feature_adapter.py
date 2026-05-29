from __future__ import annotations

import argparse
import json
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
    ) -> None:
        self.backbone = backbone
        self.adapter = adapter
        self.stage_feature_shapes = stage_feature_shapes
        self.fused_feature_shape = fused_feature_shape
        self.stage_blend = float(stage_blend)
        self.fusion_blend = float(fusion_blend)
        self.original_fuse_features = backbone.fuse_features
        self.original_forward = backbone.forward
        self.records: list[torch.Tensor] = []
        self._install()

    def clear(self) -> None:
        self.records.clear()

    def drift_loss(self, device: torch.device) -> torch.Tensor:
        if not self.records:
            return torch.zeros((), dtype=torch.float32, device=device)
        return torch.stack(self.records).mean()

    def restore(self) -> None:
        self.backbone.fuse_features = self.original_fuse_features
        self.backbone.forward = self.original_forward

    def _record(self, adapted: torch.Tensor, reference: torch.Tensor) -> None:
        self.records.append(nn.functional.smooth_l1_loss(adapted.float(), reference.detach().float()))

    def _install(self) -> None:
        backbone = self.backbone
        patch = self

        def adapted_fuse_features(image_features, lidar_features, layer_idx):
            idx = int(layer_idx)
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
            patch._record(adapted_image, image_embd_layer)
            patch._record(adapted_lidar, lidar_embd_layer)
            adapted_image = adapted_image.to(dtype=image_embd_layer.dtype)
            adapted_lidar = adapted_lidar.to(dtype=lidar_embd_layer.dtype)
            if patch.stage_blend < 1.0:
                adapted_image = image_embd_layer + patch.stage_blend * (adapted_image - image_embd_layer)
                adapted_lidar = lidar_embd_layer + patch.stage_blend * (adapted_lidar - lidar_embd_layer)

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
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                return output
            fused = output[1]
            got = tuple(int(v) for v in fused.shape[1:])
            if got != patch.fused_feature_shape:
                raise ValueError(f"fused feature shape mismatch: {got} != {patch.fused_feature_shape}")
            adapted = patch.adapter.adapt_fused(fused.float())
            patch._record(adapted, fused)
            adapted = adapted.to(dtype=fused.dtype)
            if patch.fusion_blend < 1.0:
                adapted = fused + patch.fusion_blend * (adapted - fused)
            if isinstance(output, tuple):
                return (output[0], adapted, *output[2:])
            out = list(output)
            out[1] = adapted
            return out

        backbone.fuse_features = adapted_fuse_features
        backbone.forward = adapted_forward


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


def _losses(pred: torch.Tensor, batch: Dict[str, torch.Tensor], patch: _BackboneTaskPatch, device: torch.device, args):
    target = batch["target"].to(device, non_blocking=True)
    scalar = batch["scalar"].to(device, non_blocking=True)
    stop_state = batch["stop_state"].to(device, non_blocking=True)
    stop_reason = batch["stop_reason"].to(device, non_blocking=True)
    weight = batch["sample_weight"].to(device, non_blocking=True)

    speed_dim = int(args.speed_dim)
    traj_dim = target.shape[1] - speed_dim - 1
    pred_traj = pred[:, :traj_dim].reshape(pred.shape[0], -1, 3)
    target_traj = target[:, :traj_dim].reshape(target.shape[0], -1, 3)
    pred_speed = pred[:, traj_dim : traj_dim + speed_dim]
    target_speed = target[:, traj_dim : traj_dim + speed_dim]
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
    stop = _weighted_mean(nn.functional.binary_cross_entropy_with_logits(pred[:, -1:], stop_target, reduction="none"), effective_weight)
    drift = patch.drift_loss(device)
    total = (
        float(args.xy_loss_weight) * xy
        + float(args.yaw_loss_weight) * yaw
        + float(args.speed_loss_weight) * speed
        + float(args.speed_floor_loss_weight) * speed_floor
        + float(args.launch_speed_floor_loss_weight) * launch_floor
        + float(args.release_speed_floor_loss_weight) * release_floor
        + float(args.stop_speed_ceiling_loss_weight) * stop_ceiling
        + float(args.stop_loss_weight) * stop
        + float(args.feature_drift_loss_weight) * drift
    )
    return {
        "loss": total,
        "xy": xy,
        "yaw": yaw,
        "speed": speed,
        "speed_floor": speed_floor,
        "launch_floor": launch_floor,
        "release_floor": release_floor,
        "stop_ceiling": stop_ceiling,
        "stop": stop,
        "drift": drift,
        "moving_ratio": moving.float().mean(),
        "hazard_ratio": hazard.float().mean(),
        "launch_ratio": launch.float().mean(),
        "release_ratio": release.float().mean(),
    }


def _set_frozen_tfpp_runtime_mode(net: nn.Module, train: bool) -> None:
    """Keep TF++ deterministic, but allow cuDNN RNN backward during training."""

    net.eval()
    if not train:
        return
    for module in net.modules():
        if isinstance(module, nn.modules.rnn.RNNBase):
            module.train(True)


def _run_epoch(net, adapter, patch, loader, optimizer, config, cameras, device, args, train: bool, epoch: int):
    _set_frozen_tfpp_runtime_mode(net, train)
    adapter.train(train)
    metric_names = (
        "loss",
        "xy",
        "yaw",
        "speed",
        "speed_floor",
        "launch_floor",
        "release_floor",
        "stop_ceiling",
        "stop",
        "drift",
        "moving_ratio",
        "hazard_ratio",
        "launch_ratio",
        "release_ratio",
    )
    totals = {name: 0.0 for name in metric_names}
    totals["samples"] = 0
    start = time.time()
    for step, batch in enumerate(loader, start=1):
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
        )
        patch.clear()
        with torch.set_grad_enabled(train):
            outputs = net(**inputs)
            pred_target = base_target_from_checkpoint(
                pred_checkpoint=outputs[2],
                pred_target_speed=outputs[1],
                scalar=scalar,
                config=config,
                target_dim=batch["target"].shape[1],
                speed_dim=int(args.speed_dim),
            )
            losses = _losses(pred_target, batch, patch, device, args)
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if float(args.grad_clip) > 0.0:
                    nn.utils.clip_grad_norm_(adapter.parameters(), float(args.grad_clip))
                optimizer.step()
        batch_size = int(scalar.shape[0])
        for name in metric_names:
            totals[name] += float(losses[name].detach().cpu()) * batch_size
        totals["samples"] += batch_size
        if train and int(args.step_log_every) > 0 and (step == 1 or step % int(args.step_log_every) == 0):
            samples = max(int(totals["samples"]), 1)
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"step={step:05d}/{len(loader):05d} loss={totals['loss']/samples:.6f} "
                f"xy={totals['xy']/samples:.6f} speed={totals['speed']/samples:.6f} "
                f"stopceil={totals['stop_ceiling']/samples:.6f} stop={totals['stop']/samples:.6f} "
                f"drift={totals['drift']/samples:.6f} moving={totals['moving_ratio']/samples:.3f} "
                f"samples/s={samples/elapsed:.1f}",
                flush=True,
            )
    samples = max(int(totals.pop("samples")), 1)
    return {name: value / samples for name, value in totals.items()}


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = _camera_list(args.cameras)
    train_indices, val_indices = split_by_episode(args.index, val_ratio=float(args.val_ratio), seed=int(args.seed))
    if int(args.max_train_samples) > 0:
        train_indices = train_indices[: int(args.max_train_samples)]
    if int(args.max_val_samples) > 0:
        val_indices = val_indices[: int(args.max_val_samples)]

    train_ds = Teach2DriveIndexDataset(
        args.index,
        indices=train_indices,
        cameras=cameras,
        image_size=tuple(args.image_size),
        lidar_size=int(args.lidar_size),
        episode_root_override=args.episode_root_override,
    )
    val_ds = Teach2DriveIndexDataset(
        args.index,
        indices=val_indices,
        cameras=cameras,
        image_size=tuple(args.image_size),
        lidar_size=int(args.lidar_size),
        episode_root_override=args.episode_root_override,
    )
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
    net, config, load_info = load_transfuserpp(args.garage_root, args.team_config, device=device, checkpoint=args.checkpoint)
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)

    probe_loader = DataLoader(
        train_ds,
        batch_size=max(1, min(int(args.batch_size), 2)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    stage_feature_shapes, fused_feature_shape = _infer_feature_shapes(net, config, probe_loader, cameras, args, device)
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
    patch = _BackboneTaskPatch(
        net.backbone,
        adapter,
        stage_feature_shapes,
        fused_feature_shape,
        stage_blend=float(args.stage_feature_adapter_blend),
        fusion_blend=float(args.fusion_adapter_blend),
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    metadata = {
        "mode": "transfuserpp_extrinsic_task_feature_then_fusion_adapter" if args.extrinsic_aware else "transfuserpp_task_feature_then_fusion_adapter",
        "index": str(Path(args.index).expanduser()),
        "episode_root_override": str(Path(args.episode_root_override).expanduser()) if args.episode_root_override else "",
        "garage_root": str(Path(args.garage_root).expanduser()),
        "team_config": str(Path(args.team_config).expanduser()),
        "tfpp_load_info": load_info,
        "adapter_init_checkpoint": str(Path(args.init_checkpoint).expanduser()) if args.init_checkpoint else "",
        "adapter_init_load_info": load_adapter_info,
        "cameras": cameras,
        "tfpp_camera": args.tfpp_camera,
        "command_mode": args.command_mode,
        "image_size": list(args.image_size),
        "lidar_size": int(args.lidar_size),
        "stage_feature_shapes": {key: list(value) for key, value in stage_feature_shapes.items()},
        "fused_feature_shape": list(fused_feature_shape),
        "extrinsic_aware": bool(args.extrinsic_aware),
        "source_profile": args.source_profile,
        "extrinsic_vector": [float(v) for v in extrinsic_vector],
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "args": vars(args),
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
                "tfpp_load_info": load_info,
                "adapter_init_load_info": load_adapter_info,
            },
            indent=2,
        ),
        flush=True,
    )

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    try:
        for epoch in range(1, int(args.epochs) + 1):
            train_metrics = _run_epoch(net, adapter, patch, train_loader, optimizer, config, cameras, device, args, train=True, epoch=epoch)
            val_metrics = _run_epoch(net, adapter, patch, val_loader, optimizer, config, cameras, device, args, train=False, epoch=epoch)
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
            val_loss = float(val_metrics["loss"])
            print(
                f"epoch={epoch:03d} train={train_metrics['loss']:.6f} val={val_loss:.6f} "
                f"best={best_val if best_epoch else val_loss:.6f} "
                f"xy={val_metrics['xy']:.6f} speed={val_metrics['speed']:.6f} drift={val_metrics['drift']:.6f}",
                flush=True,
            )
            if val_loss + float(args.early_stop_min_delta) < best_val:
                best_val = val_loss
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model_state": adapter.state_dict(),
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
                    print(
                        f"early_stop: no val improvement for {stale} epochs "
                        f"(patience={args.early_stop_patience}, best={best_val:.6f})",
                        flush=True,
                    )
                    break
    finally:
        patch.restore()

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {"best_epoch": int(best_epoch), "best_val_loss": float(best_val), "mode": metadata["mode"]}
    if (out_dir / "best_model.pt").exists():
        best = torch.load(out_dir / "best_model.pt", map_location="cpu")
        summary.update(best.get("val_metrics", {}))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a target-only task-driven TransFuser++ feature-then-fusion adapter.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--episode-root-override", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--garage-root", required=True)
    parser.add_argument("--team-config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--cameras", default="left,front,right")
    parser.add_argument("--tfpp-camera", default="front")
    parser.add_argument("--command-mode", choices=["lane_follow", "target_angle"], default="target_angle")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--lidar-size", type=int, default=128)
    parser.add_argument("--extrinsic-aware", action="store_true")
    parser.add_argument("--source-profile", default="front_triplet_shifted", choices=["front_triplet_shifted", "tfpp_ego"])
    parser.add_argument("--extrinsic-hidden-dim", type=int, default=64)
    parser.add_argument("--extrinsic-dropout", type=float, default=0.0)
    parser.add_argument("--hidden-channels", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--stage-feature-adapter-blend", type=float, default=1.0)
    parser.add_argument("--fusion-adapter-blend", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--speed-dim", type=int, default=4)
    parser.add_argument("--xy-loss-weight", type=float, default=0.55)
    parser.add_argument("--yaw-loss-weight", type=float, default=0.03)
    parser.add_argument("--speed-loss-weight", type=float, default=0.80)
    parser.add_argument("--speed-floor-loss-weight", type=float, default=0.03)
    parser.add_argument("--speed-floor-mps", type=float, default=0.8)
    parser.add_argument("--speed-floor-target-threshold", type=float, default=2.0)
    parser.add_argument("--stop-speed-ceiling-loss-weight", type=float, default=0.35)
    parser.add_argument("--stop-speed-ceiling-mps", type=float, default=0.5)
    parser.add_argument("--stop-speed-target-threshold", type=float, default=0.5)
    parser.add_argument("--stop-loss-weight", type=float, default=0.08)
    parser.add_argument("--feature-drift-loss-weight", type=float, default=0.10)
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
