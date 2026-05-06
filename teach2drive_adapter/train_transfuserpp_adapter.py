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
from .model import count_trainable_parameters
from .train_adapter import _masked_weighted_ce, _weighted_mean
from .transfuserpp_adapter_model import TransFuserPPResidualAdapterPolicy


def _camera_list(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def _substring_list(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def _adapter_state_dict(model: nn.Module):
    return {key: value for key, value in model.state_dict().items() if not key.startswith("net.")}


def _build_optimizer(model: nn.Module, args: argparse.Namespace):
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    prior_params = [param for param in raw_model.net.parameters() if param.requires_grad]
    adapter_params = [param for name, param in raw_model.named_parameters() if not name.startswith("net.") and param.requires_grad]
    param_groups = []
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": args.lr, "weight_decay": args.weight_decay, "group_name": "adapter"})
    if prior_params:
        param_groups.append({"params": prior_params, "lr": args.prior_lr, "weight_decay": args.prior_weight_decay, "group_name": "prior"})
    if not param_groups:
        raise ValueError("No trainable parameters were found for optimizer setup.")
    optimizer = torch.optim.AdamW(param_groups)
    return optimizer, {
        "adapter_parameters": int(sum(param.numel() for param in adapter_params)),
        "prior_parameters": int(sum(param.numel() for param in prior_params)),
        "adapter_lr": float(args.lr),
        "prior_lr": float(args.prior_lr if prior_params else 0.0),
        "prior_weight_decay": float(args.prior_weight_decay if prior_params else 0.0),
    }


def _run_epoch(model, loader, optimizer, device, args, train: bool) -> Dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "xy": 0.0, "yaw": 0.0, "speed": 0.0, "stop": 0.0, "state": 0.0, "reason": 0.0, "samples": 0}
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        scalar = batch["scalar"].to(device, non_blocking=True)
        camera = batch["camera"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        layout = batch["layout"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        stop_state = batch["stop_state"].to(device, non_blocking=True)
        stop_reason = batch["stop_reason"].to(device, non_blocking=True)
        stop_reason_mask = batch["stop_reason_mask"].to(device, non_blocking=True)
        weight = batch["sample_weight"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            out = model(scalar, camera, lidar, layout)
            pred = out["target"]
            speed_dim = int(args.speed_dim)
            traj_dim = pred.shape[1] - speed_dim - 1
            pred_traj = pred[:, :traj_dim].reshape(pred.shape[0], -1, 3)
            target_traj = target[:, :traj_dim].reshape(target.shape[0], -1, 3)
            xy_loss = _weighted_mean(torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., :2], target_traj[..., :2], reduction="none"), dim=(1, 2)), weight)
            yaw_loss = _weighted_mean(torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., 2], target_traj[..., 2], reduction="none"), dim=1), weight)
            speed_loss = _weighted_mean(
                torch.mean(nn.functional.smooth_l1_loss(pred[:, traj_dim : traj_dim + speed_dim], target[:, traj_dim : traj_dim + speed_dim], reduction="none"), dim=1),
                weight,
            )
            stop_loss = _weighted_mean(
                nn.functional.binary_cross_entropy_with_logits(pred[:, -1:], target[:, -1:], reduction="none"),
                weight,
            )
            state_loss = _weighted_mean(nn.functional.cross_entropy(out["stop_state"], stop_state, reduction="none"), weight)
            reason_loss = _masked_weighted_ce(out["stop_reason"], stop_reason, stop_reason_mask, weight)
            loss = (
                args.xy_loss_weight * xy_loss
                + args.yaw_loss_weight * yaw_loss
                + args.speed_loss_weight * speed_loss
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
        totals["xy"] += float(xy_loss.detach().cpu()) * batch_size
        totals["yaw"] += float(yaw_loss.detach().cpu()) * batch_size
        totals["speed"] += float(speed_loss.detach().cpu()) * batch_size
        totals["stop"] += float(stop_loss.detach().cpu()) * batch_size
        totals["state"] += float(state_loss.detach().cpu()) * batch_size
        totals["reason"] += float(reason_loss.detach().cpu()) * batch_size
        totals["samples"] += batch_size
        if train and args.step_log_every > 0 and (step == 1 or step % args.step_log_every == 0):
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"step={step:05d}/{len(loader):05d} loss={totals['loss']/totals['samples']:.6f} "
                f"xy={totals['xy']/totals['samples']:.6f} samples/s={totals['samples']/elapsed:.1f}",
                flush=True,
            )

    samples = max(int(totals.pop("samples")), 1)
    return {key: value / samples for key, value in totals.items()}


def _evaluate_predictions_with_layout(model, loader, device) -> Dict:
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
            layout = batch["layout"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            stop_state = batch["stop_state"].to(device, non_blocking=True)
            stop_reason = batch["stop_reason"].to(device, non_blocking=True)
            stop_reason_mask = batch["stop_reason_mask"].to(device, non_blocking=True).reshape(-1).bool()
            out = model(scalar, camera, lidar, layout)
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
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    train_indices, val_indices = split_by_episode(args.index, val_ratio=args.val_ratio, seed=args.seed)
    if args.max_train_samples > 0:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_indices = val_indices[: args.max_val_samples]

    cameras = _camera_list(args.cameras)
    train_ds = Teach2DriveIndexDataset(
        args.index,
        indices=train_indices,
        cameras=cameras,
        image_size=tuple(args.image_size),
        lidar_size=args.lidar_size,
        episode_root_override=args.episode_root_override,
    )
    val_ds = Teach2DriveIndexDataset(
        args.index,
        indices=val_indices,
        cameras=cameras,
        image_size=tuple(args.image_size),
        lidar_size=args.lidar_size,
        episode_root_override=args.episode_root_override,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = TransFuserPPResidualAdapterPolicy(
        garage_root=args.garage_root,
        team_config=args.team_config,
        checkpoint=args.checkpoint,
        device=device,
        scalar_dim=train_ds.scalar_dim,
        target_dim=train_ds.target_dim + 1,
        layout_dim=train_ds.layout_dim,
        cameras=cameras,
        hidden_dim=args.hidden_dim,
        speed_dim=train_ds.speed_dim,
        command_mode=args.command_mode,
        tfpp_camera=args.tfpp_camera,
        layout_hidden_dim=args.layout_hidden_dim,
        train_prior_substrings=_substring_list(args.train_prior_substrings),
    )
    model.to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    optimizer, optimizer_info = _build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    print(
        json.dumps(
            {
                "mode": "transfuserpp_residual_adapter",
                "parameters": count_trainable_parameters(raw_model),
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "cameras": cameras,
                "tfpp_camera": args.tfpp_camera,
                "command_mode": args.command_mode,
                "transfuserpp_load_info": raw_model.load_info,
                "prior_trainable": raw_model.prior_trainable_info(),
                "optimizer": optimizer_info,
            },
            indent=2,
        ),
        flush=True,
    )

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_metrics = _run_epoch(model, val_loader, optimizer, device, args, train=False)
        scheduler.step()
        row = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]), "train": train_metrics, "val": val_metrics, "best_val_loss": best_val}
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            row["best_val_loss"] = best_val
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "model_state": _adapter_state_dict(raw_model),
                    "args": vars(args),
                    "scalar_dim": train_ds.scalar_dim,
                    "target_dim": train_ds.target_dim + 1,
                    "layout_dim": train_ds.layout_dim,
                    "num_cameras": len(train_ds.cameras),
                    "cameras": cameras,
                    "epoch": epoch,
                    "val_loss": best_val,
                    "transfuserpp_load_info": raw_model.load_info,
                },
                out_dir / "best_model.pt",
            )
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (out_dir / "latest.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"epoch={epoch:03d} train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f} best={best_val:.6f}", flush=True)

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    best = torch.load(out_dir / "best_model.pt", map_location="cpu")
    raw_model.load_state_dict(best["model_state"], strict=False)
    raw_model.to(device)
    metrics = _evaluate_predictions_with_layout(raw_model, val_loader, device)
    metrics.update({"best_epoch": int(best["epoch"]), "best_val_loss": float(best["val_loss"]), "mode": "transfuserpp_residual_adapter"})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small layout-aware adapter on top of frozen CARLA Garage TransFuser++.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--garage-root", required=True)
    parser.add_argument("--team-config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--episode-root-override", default="")
    parser.add_argument("--cameras", default="front,left,right")
    parser.add_argument("--tfpp-camera", default="front")
    parser.add_argument("--command-mode", choices=["lane_follow", "target_angle"], default="lane_follow")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prior-lr", type=float, default=1e-5)
    parser.add_argument("--prior-weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layout-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--train-prior-substrings",
        default="",
        help="Comma-separated TransFuser++ parameter-name substrings to unfreeze. Leave empty for adapter-only training.",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--lidar-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--speed-dim", type=int, default=4)
    parser.add_argument("--xy-loss-weight", type=float, default=1.0)
    parser.add_argument("--yaw-loss-weight", type=float, default=0.05)
    parser.add_argument("--speed-loss-weight", type=float, default=0.10)
    parser.add_argument("--stop-loss-weight", type=float, default=0.05)
    parser.add_argument("--stop-state-loss-weight", type=float, default=0.10)
    parser.add_argument("--stop-reason-loss-weight", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--step-log-every", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
