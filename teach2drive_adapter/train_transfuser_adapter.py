import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import Teach2DriveIndexDataset, split_by_episode
from .model import count_trainable_parameters
from .train_adapter import _evaluate_predictions, _masked_weighted_ce, _weighted_mean
from .transfuser_adapter_model import TransFuserResidualAdapterPolicy


def _camera_list(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def _adapter_state_dict(model: nn.Module):
    return {key: value for key, value in model.state_dict().items() if not key.startswith("net.")}


def _run_epoch(model, loader, optimizer, device, args, train: bool):
    model.train(train)
    totals = {
        "loss": 0.0,
        "xy": 0.0,
        "yaw": 0.0,
        "speed": 0.0,
        "stop": 0.0,
        "state": 0.0,
        "reason": 0.0,
        "samples": 0,
    }
    import time

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
            pred = out["target"]
            target_dim = pred.shape[1]
            speed_dim = int(args.speed_dim)
            traj_dim = target_dim - speed_dim - 1
            pred_traj = pred[:, :traj_dim].reshape(pred.shape[0], -1, 3)
            target_traj = target[:, :traj_dim].reshape(target.shape[0], -1, 3)
            xy_loss = _weighted_mean(torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., :2], target_traj[..., :2], reduction="none"), dim=(1, 2)), weight)
            yaw_loss = _weighted_mean(torch.mean(nn.functional.smooth_l1_loss(pred_traj[..., 2], target_traj[..., 2], reduction="none"), dim=1), weight)
            speed_loss = _weighted_mean(
                torch.mean(nn.functional.smooth_l1_loss(pred[:, traj_dim : traj_dim + speed_dim], target[:, traj_dim : traj_dim + speed_dim], reduction="none"), dim=1),
                weight,
            )
            stop_loss = _weighted_mean(
                nn.functional.binary_cross_entropy_with_logits(pred[:, target_dim - 1 : target_dim], target[:, target_dim - 1 : target_dim], reduction="none"),
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


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    train_indices, val_indices = split_by_episode(args.index, val_ratio=args.val_ratio, seed=args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    if args.max_train_samples > 0:
        perm = torch.randperm(len(train_indices), generator=rng).numpy()
        train_indices = train_indices[perm[: args.max_train_samples]]
    if args.max_val_samples > 0:
        perm = torch.randperm(len(val_indices), generator=rng).numpy()
        val_indices = val_indices[perm[: args.max_val_samples]]

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

    model = TransFuserResidualAdapterPolicy(
        transfuser_root=args.transfuser_root,
        team_config=args.team_config,
        device=device,
        scalar_dim=train_ds.scalar_dim,
        target_dim=train_ds.target_dim + 1,
        hidden_dim=args.hidden_dim,
        speed_dim=train_ds.speed_dim,
    )
    model.to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    parameter_counts = count_trainable_parameters(raw_model)
    init_val_loss = _run_epoch(model, val_loader, optimizer, device, args, train=False)
    init_metrics = _evaluate_predictions(model, val_loader, device)
    print(
        json.dumps(
            {
                "mode": "transfuser_residual_adapter",
                "parameters": parameter_counts,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "cameras": cameras,
                "transfuser_load_info": raw_model.load_info,
                "initial_val_loss": init_val_loss,
                "initial_val_metrics": init_metrics,
            },
            indent=2,
        ),
        flush=True,
    )
    (out_dir / "initial_loss.json").write_text(json.dumps(init_val_loss, indent=2), encoding="utf-8")
    (out_dir / "initial_metrics.json").write_text(json.dumps(init_metrics, indent=2), encoding="utf-8")

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
                    "model_state": _adapter_state_dict(raw_model),
                    "args": vars(args),
                    "scalar_dim": train_ds.scalar_dim,
                    "target_dim": train_ds.target_dim + 1,
                    "num_cameras": len(train_ds.cameras),
                    "cameras": cameras,
                    "epoch": epoch,
                    "val_loss": best_val,
                    "transfuser_load_info": raw_model.load_info,
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
    metrics = _evaluate_predictions(raw_model, val_loader, device)
    metrics.update(
        {
            "best_epoch": int(best["epoch"]),
            "best_val_loss": float(best["val_loss"]),
            "mode": "transfuser_residual_adapter",
            "initial_mean_xy_error_m_by_horizon": init_metrics.get("mean_xy_error_m_by_horizon"),
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small Teach2Drive residual adapter on top of frozen official TransFuser weights.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--transfuser-root", default="/home/byeongjae/code/transfuser")
    parser.add_argument("--team-config", default="/home/byeongjae/code/transfuser/model_ckpt/models_2022/transfuser")
    parser.add_argument("--episode-root-override", default="")
    parser.add_argument("--cameras", default="left,front,right")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--lidar-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--speed-dim", type=int, default=4)
    parser.add_argument("--xy-loss-weight", type=float, default=1.0)
    parser.add_argument("--yaw-loss-weight", type=float, default=0.05)
    parser.add_argument("--speed-loss-weight", type=float, default=0.05)
    parser.add_argument("--stop-loss-weight", type=float, default=0.0)
    parser.add_argument("--stop-state-loss-weight", type=float, default=0.0)
    parser.add_argument("--stop-reason-loss-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--step-log-every", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
