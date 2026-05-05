import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import STOP_REASON_NAMES, STOP_STATE_NAMES, Teach2DriveIndexDataset, split_by_episode
from .model import Teach2DriveAdapterPolicy, configure_train_mode, count_trainable_parameters


def _weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.reshape(-1)
    return torch.sum(loss.reshape(-1) * weight) / torch.clamp(torch.sum(weight), min=1e-6)


def _masked_weighted_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    active = mask.reshape(-1) * weight.reshape(-1)
    loss = nn.functional.cross_entropy(logits, target.long(), reduction="none")
    return torch.sum(loss * active) / torch.clamp(torch.sum(active), min=1e-6)


def _run_epoch(model, loader, optimizer, device, args, train: bool) -> Dict[str, float]:
    model.train(train)
    target_loss_fn = nn.MSELoss(reduction="none")
    stop_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    totals = {"loss": 0.0, "target": 0.0, "stop": 0.0, "state": 0.0, "reason": 0.0, "samples": 0}
    start = time.time()

    for step, batch in enumerate(loader, start=1):
        scalar = batch["scalar"].to(device, non_blocking=True)
        camera = batch["camera"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        stop_state = batch["stop_state"].to(device, non_blocking=True)
        stop_reason = batch["stop_reason"].to(device, non_blocking=True)
        stop_reason_mask = batch["stop_reason_mask"].to(device, non_blocking=True)
        weight = batch["sample_weight"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            out = model(scalar, camera, lidar)
            target_pred = out["target"]
            target_dim = target_pred.shape[1]
            traj_speed_loss = _weighted_mean(torch.mean(target_loss_fn(target_pred[:, : target_dim - 1], target[:, : target_dim - 1]), dim=1), weight)
            stop_loss = _weighted_mean(stop_loss_fn(target_pred[:, target_dim - 1 : target_dim], target[:, target_dim - 1 : target_dim]), weight)
            state_loss = _weighted_mean(nn.functional.cross_entropy(out["stop_state"], stop_state, reduction="none"), weight)
            reason_loss = _masked_weighted_ce(out["stop_reason"], stop_reason, stop_reason_mask, weight)
            loss = (
                args.target_loss_weight * traj_speed_loss
                + args.stop_loss_weight * stop_loss
                + args.stop_state_loss_weight * state_loss
                + args.stop_reason_loss_weight * reason_loss
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        batch_size = int(scalar.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["target"] += float(traj_speed_loss.detach().cpu()) * batch_size
        totals["stop"] += float(stop_loss.detach().cpu()) * batch_size
        totals["state"] += float(state_loss.detach().cpu()) * batch_size
        totals["reason"] += float(reason_loss.detach().cpu()) * batch_size
        totals["samples"] += batch_size
        if train and args.step_log_every > 0 and (step == 1 or step % args.step_log_every == 0):
            elapsed = max(time.time() - start, 1e-6)
            print(f"step={step:05d}/{len(loader):05d} loss={totals['loss']/totals['samples']:.6f} samples/s={totals['samples']/elapsed:.1f}", flush=True)

    samples = max(int(totals.pop("samples")), 1)
    return {key: value / samples for key, value in totals.items()}


def _evaluate_predictions(model, loader, device) -> Dict:
    model.eval()
    stop_correct = 0
    stop_total = 0
    state_correct = 0
    state_total = 0
    reason_correct = 0
    reason_total = 0
    xy_errors = []
    with torch.no_grad():
        for batch in loader:
            scalar = batch["scalar"].to(device, non_blocking=True)
            camera = batch["camera"].to(device, non_blocking=True)
            lidar = batch["lidar"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            stop_state = batch["stop_state"].to(device, non_blocking=True)
            stop_reason = batch["stop_reason"].to(device, non_blocking=True)
            stop_reason_mask = batch["stop_reason_mask"].to(device, non_blocking=True).reshape(-1).bool()
            out = model(scalar, camera, lidar)
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
            traj_dim = pred.shape[1] - 1 - 4
            if traj_dim > 0 and traj_dim % 3 == 0:
                pred_xy = pred[:, :traj_dim].reshape(pred.shape[0], -1, 3)[..., :2]
                target_xy = target[:, :traj_dim].reshape(target.shape[0], -1, 3)[..., :2]
                xy_errors.append(torch.linalg.norm(pred_xy - target_xy, dim=-1).mean(dim=0).cpu().numpy())
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
    return metrics


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_indices, val_indices = split_by_episode(args.index, val_ratio=args.val_ratio, seed=args.seed)
    if args.max_train_samples > 0:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_indices = val_indices[: args.max_val_samples]
    train_ds = Teach2DriveIndexDataset(
        args.index,
        indices=train_indices,
        image_size=tuple(args.image_size),
        lidar_size=args.lidar_size,
        episode_root_override=args.episode_root_override,
    )
    val_ds = Teach2DriveIndexDataset(
        args.index,
        indices=val_indices,
        image_size=tuple(args.image_size),
        lidar_size=args.lidar_size,
        episode_root_override=args.episode_root_override,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    sample = train_ds[0]
    lidar_channels = int(sample["lidar"].shape[0])
    model = Teach2DriveAdapterPolicy(
        scalar_dim=train_ds.scalar_dim,
        num_cameras=len(train_ds.cameras),
        target_dim=train_ds.target_dim + 1,
        embed_dim=args.embed_dim,
        adapter_dim=args.adapter_dim,
        lidar_channels=lidar_channels,
    )
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded_pretrained={args.pretrained} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    configure_train_mode(model, args.mode)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    parameter_counts = count_trainable_parameters(model.module if isinstance(model, nn.DataParallel) else model)
    print(json.dumps({"mode": args.mode, "parameters": parameter_counts, "train_samples": len(train_ds), "val_samples": len(val_ds)}, indent=2), flush=True)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_metrics = _run_epoch(model, val_loader, optimizer, device, args, train=False)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
            "best_val_loss": best_val,
        }
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            row["best_val_loss"] = best_val
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "model_state": raw_model.state_dict(),
                    "args": vars(args),
                    "scalar_dim": train_ds.scalar_dim,
                    "target_dim": train_ds.target_dim + 1,
                    "num_cameras": len(train_ds.cameras),
                    "lidar_channels": lidar_channels,
                    "epoch": epoch,
                    "val_loss": best_val,
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
    metrics = _evaluate_predictions(raw_model, val_loader, device)
    metrics.update({"best_epoch": int(best["epoch"]), "best_val_loss": float(best["val_loss"]), "mode": args.mode})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Teach2Drive adapter on a unified token index.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--mode", choices=["scratch", "adapter", "head", "full"], default="adapter")
    parser.add_argument("--episode-root-override", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--image-size", type=int, nargs=2, default=[320, 180], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--lidar-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Debug option. Use 0 for all training samples.")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Debug option. Use 0 for all validation samples.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--stop-loss-weight", type=float, default=0.15)
    parser.add_argument("--stop-state-loss-weight", type=float, default=0.35)
    parser.add_argument("--stop-reason-loss-weight", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--step-log-every", type=int, default=200)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
