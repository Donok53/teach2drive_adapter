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
from .layout_conditioning import FiLMLayoutAdapter
from .model import count_trainable_parameters
from .train_adapter import _masked_weighted_ce, _weighted_mean
from .transfuserpp_adapter_model import TransFuserPPResidualHeads


def _split_by_episode(sample_episode: np.ndarray, val_ratio: float, seed: int):
    episodes = np.unique(sample_episode)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * val_ratio)))
    val_episodes = set(int(v) for v in episodes[:val_count])
    mask = np.asarray([int(ep) in val_episodes for ep in sample_episode], dtype=bool)
    ids = np.arange(len(sample_episode), dtype=np.int64)
    return ids[~mask], ids[mask]


class CachedPriorDataset(Dataset):
    def __init__(self, cache_path: str, indices: Optional[np.ndarray] = None) -> None:
        self.cache_path = Path(cache_path).expanduser()
        arrays = np.load(self.cache_path, allow_pickle=True)
        self.sample_episode = arrays["sample_episode"].astype(np.int64)
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
        self.metadata = json.loads(str(arrays["metadata"].item())) if "metadata" in arrays.files else {}
        self.indices = np.arange(len(self.scalar), dtype=np.int64) if indices is None else indices.astype(np.int64)

    @property
    def scalar_dim(self) -> int:
        return int(self.scalar.shape[1])

    @property
    def layout_dim(self) -> int:
        return int(self.layout.shape[1])

    @property
    def target_dim(self) -> int:
        return int(self.target.shape[1])

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        return {
            "scalar": torch.from_numpy(self.scalar[idx]),
            "layout": torch.from_numpy(self.layout[idx]),
            "target": torch.from_numpy(self.target[idx]),
            "stop_state": torch.tensor(self.stop_state[idx], dtype=torch.long),
            "stop_reason": torch.tensor(self.stop_reason[idx], dtype=torch.long),
            "stop_reason_mask": torch.from_numpy(self.stop_reason_mask[idx]),
            "sample_weight": torch.from_numpy(self.sample_weight[idx]),
            "base_target": torch.from_numpy(self.base_target[idx]),
            "checkpoint_flat": torch.from_numpy(self.checkpoint_flat[idx]),
            "speed_logits": torch.from_numpy(self.speed_logits[idx]),
            "expected_speed": torch.from_numpy(self.expected_speed[idx]),
        }


class CachedTransFuserPPAdapterPolicy(nn.Module):
    def __init__(self, scalar_dim: int, layout_dim: int, target_dim: int, checkpoint_dim: int, speed_classes: int, hidden_dim: int = 512, layout_hidden_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = scalar_dim + layout_dim + target_dim + checkpoint_dim + speed_classes + 1
        self.layout_adapter = FiLMLayoutAdapter(feature_dim=self.feature_dim, layout_dim=layout_dim, hidden_dim=layout_hidden_dim)
        self.heads = TransFuserPPResidualHeads(self.feature_dim, target_dim, hidden_dim=hidden_dim)

    def forward(self, scalar: torch.Tensor, layout: torch.Tensor, base_target: torch.Tensor, checkpoint_flat: torch.Tensor, speed_logits: torch.Tensor, expected_speed: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = torch.cat([scalar, layout, base_target, checkpoint_flat, speed_logits, expected_speed], dim=1)
        adapted = self.layout_adapter(features, layout)["features"]
        return self.heads(adapted, base_target)


def _run_epoch(model, loader, optimizer, device, args, train: bool) -> Dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "xy": 0.0, "yaw": 0.0, "speed": 0.0, "stop": 0.0, "state": 0.0, "reason": 0.0, "samples": 0}
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        scalar = batch["scalar"].to(device, non_blocking=True)
        layout = batch["layout"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        base_target = batch["base_target"].to(device, non_blocking=True)
        checkpoint_flat = batch["checkpoint_flat"].to(device, non_blocking=True)
        speed_logits = batch["speed_logits"].to(device, non_blocking=True)
        expected_speed = batch["expected_speed"].to(device, non_blocking=True)
        stop_state = batch["stop_state"].to(device, non_blocking=True)
        stop_reason = batch["stop_reason"].to(device, non_blocking=True)
        stop_reason_mask = batch["stop_reason_mask"].to(device, non_blocking=True)
        weight = batch["sample_weight"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = model(scalar, layout, base_target, checkpoint_flat, speed_logits, expected_speed)
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
            stop_loss = _weighted_mean(nn.functional.binary_cross_entropy_with_logits(pred[:, -1:], target[:, -1:], reduction="none"), weight)
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
            print(f"step={step:05d}/{len(loader):05d} loss={totals['loss']/totals['samples']:.6f} xy={totals['xy']/totals['samples']:.6f} samples/s={totals['samples']/elapsed:.1f}", flush=True)
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
            layout = batch["layout"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            base_target = batch["base_target"].to(device, non_blocking=True)
            checkpoint_flat = batch["checkpoint_flat"].to(device, non_blocking=True)
            speed_logits = batch["speed_logits"].to(device, non_blocking=True)
            expected_speed = batch["expected_speed"].to(device, non_blocking=True)
            stop_state = batch["stop_state"].to(device, non_blocking=True)
            stop_reason = batch["stop_reason"].to(device, non_blocking=True)
            stop_reason_mask = batch["stop_reason_mask"].to(device, non_blocking=True).reshape(-1).bool()
            out = model(scalar, layout, base_target, checkpoint_flat, speed_logits, expected_speed)
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
    full_ds = CachedPriorDataset(args.cache)
    train_indices, val_indices = _split_by_episode(full_ds.sample_episode, val_ratio=args.val_ratio, seed=args.seed)
    if args.max_train_samples > 0:
        train_indices = train_indices[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_indices = val_indices[: args.max_val_samples]
    train_ds = CachedPriorDataset(args.cache, train_indices)
    val_ds = CachedPriorDataset(args.cache, val_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = CachedTransFuserPPAdapterPolicy(
        scalar_dim=train_ds.scalar_dim,
        layout_dim=train_ds.layout_dim,
        target_dim=train_ds.target_dim,
        checkpoint_dim=int(train_ds.checkpoint_flat.shape[1]),
        speed_classes=int(train_ds.speed_logits.shape[1]),
        hidden_dim=args.hidden_dim,
        layout_hidden_dim=args.layout_hidden_dim,
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
                "mode": "transfuserpp_cached_residual_adapter",
                "parameters": count_trainable_parameters(raw_model),
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "cache": str(Path(args.cache).expanduser()),
                "cache_metadata": train_ds.metadata,
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
                    "model_state": raw_model.state_dict(),
                    "args": vars(args),
                    "scalar_dim": train_ds.scalar_dim,
                    "layout_dim": train_ds.layout_dim,
                    "target_dim": train_ds.target_dim,
                    "checkpoint_dim": int(train_ds.checkpoint_flat.shape[1]),
                    "speed_classes": int(train_ds.speed_logits.shape[1]),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "cache_metadata": train_ds.metadata,
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
    metrics.update({"best_epoch": int(best["epoch"]), "best_val_loss": float(best["val_loss"]), "mode": "transfuserpp_cached_residual_adapter"})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a fast cached layout adapter from frozen TransFuser++ prior features.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layout-hidden-dim", type=int, default=128)
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

