import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _load_metadata(cache_dir: Path) -> Dict:
    path = cache_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature cache metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _split_by_episode(sample_episode: np.ndarray, val_ratio: float, seed: int):
    episodes = np.unique(sample_episode.astype(np.int64))
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * val_ratio)))
    val_episodes = set(int(v) for v in episodes[:val_count])
    mask = np.asarray([int(ep) in val_episodes for ep in sample_episode], dtype=bool)
    indices = np.arange(len(sample_episode), dtype=np.int64)
    return indices[~mask], indices[mask]


class FusedFeatureCache:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.metadata = _load_metadata(self.cache_dir)
        self.features = np.load(self.cache_dir / "fused_features.npy", mmap_mode="r")
        self.sample_episode = np.load(self.cache_dir / "sample_episode.npy", mmap_mode="r")
        self.sample_frame = np.load(self.cache_dir / "sample_frame.npy", mmap_mode="r")
        self.sample_index = np.load(self.cache_dir / "sample_index.npy", mmap_mode="r")
        if len(self.features) != len(self.sample_episode):
            raise ValueError(f"cache length mismatch in {self.cache_dir}")


class FusedFeaturePairDataset(Dataset):
    def __init__(self, source: FusedFeatureCache, target: FusedFeatureCache, indices: Optional[np.ndarray] = None) -> None:
        if source.features.shape != target.features.shape:
            raise ValueError(f"feature shape mismatch: {source.features.shape} != {target.features.shape}")
        if not np.array_equal(source.sample_episode, target.sample_episode):
            raise ValueError("source/target feature caches are not aligned by episode")
        if not np.array_equal(source.sample_frame, target.sample_frame):
            raise ValueError("source/target feature caches are not aligned by frame")
        self.source = source
        self.target = target
        self.indices = np.arange(len(source.features), dtype=np.int64) if indices is None else indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        return {
            "source": torch.from_numpy(np.array(self.source.features[idx], copy=True)),
            "target": torch.from_numpy(np.array(self.target.features[idx], copy=True)),
            "episode": torch.tensor(int(self.source.sample_episode[idx]), dtype=torch.long),
            "frame": torch.tensor(int(self.source.sample_frame[idx]), dtype=torch.long),
        }


def _group_count(channels: int) -> int:
    groups = min(32, int(channels))
    while groups > 1 and int(channels) % groups != 0:
        groups -= 1
    return groups


class ResidualFusedFeatureAdapter(nn.Module):
    """Small residual Conv adapter from shifted fused features to canonical fused features."""

    def __init__(self, channels: int, hidden_channels: int = 0, blocks: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(hidden_channels) if hidden_channels > 0 else max(64, int(channels) // 2)
        layers = []
        in_channels = int(channels)
        final_convs = []
        for _ in range(int(blocks)):
            layers.extend(
                [
                    nn.GroupNorm(_group_count(in_channels), in_channels),
                    nn.Conv2d(in_channels, hidden, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
            if float(dropout) > 0.0:
                layers.append(nn.Dropout2d(float(dropout)))
            final_conv = nn.Conv2d(hidden, in_channels, kernel_size=1)
            layers.append(final_conv)
            final_convs.append(final_conv)
        self.net = nn.Sequential(*layers)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        for module in final_convs:
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.net(x)


def _feature_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_pool = pred.mean(dim=(2, 3))
    target_pool = target.mean(dim=(2, 3))
    return 1.0 - nn.functional.cosine_similarity(pred_pool, target_pool, dim=1).mean()


def _run_epoch(model, loader, optimizer, device, args, train: bool) -> Dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "feature": 0.0, "cosine": 0.0, "residual": 0.0, "input_l1": 0.0, "adapted_l1": 0.0, "samples": 0}
    start = time.time()
    for step, batch in enumerate(loader, start=1):
        source = batch["source"].to(device, non_blocking=True).float()
        target = batch["target"].to(device, non_blocking=True).float()
        with torch.set_grad_enabled(train):
            pred = model(source)
            feature_loss = nn.functional.smooth_l1_loss(pred, target)
            cosine_loss = _feature_cosine_loss(pred, target)
            residual_loss = torch.mean(torch.abs(pred - source))
            loss = (
                float(args.feature_loss_weight) * feature_loss
                + float(args.cosine_loss_weight) * cosine_loss
                + float(args.residual_loss_weight) * residual_loss
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(args.grad_clip) > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()
        batch_size = int(source.shape[0])
        input_l1 = torch.mean(torch.abs(source - target))
        adapted_l1 = torch.mean(torch.abs(pred.detach() - target))
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["feature"] += float(feature_loss.detach().cpu()) * batch_size
        totals["cosine"] += float(cosine_loss.detach().cpu()) * batch_size
        totals["residual"] += float(residual_loss.detach().cpu()) * batch_size
        totals["input_l1"] += float(input_l1.detach().cpu()) * batch_size
        totals["adapted_l1"] += float(adapted_l1.detach().cpu()) * batch_size
        totals["samples"] += batch_size
        if train and int(args.step_log_every) > 0 and (step == 1 or step % int(args.step_log_every) == 0):
            samples = max(totals["samples"], 1)
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"step={step:05d}/{len(loader):05d} "
                f"loss={totals['loss']/samples:.6f} "
                f"feature={totals['feature']/samples:.6f} "
                f"cos={totals['cosine']/samples:.6f} "
                f"residual={totals['residual']/samples:.6f} "
                f"input_l1={totals['input_l1']/samples:.6f} "
                f"adapted_l1={totals['adapted_l1']/samples:.6f} "
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
    train_indices, val_indices = _split_by_episode(source_cache.sample_episode, float(args.val_ratio), int(args.seed))
    if int(args.max_train_samples) > 0:
        train_indices = train_indices[: int(args.max_train_samples)]
    if int(args.max_val_samples) > 0:
        val_indices = val_indices[: int(args.max_val_samples)]
    train_ds = FusedFeaturePairDataset(source_cache, target_cache, train_indices)
    val_ds = FusedFeaturePairDataset(source_cache, target_cache, val_indices)
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
    feature_shape = tuple(int(v) for v in source_cache.features.shape[1:])
    if len(feature_shape) != 3:
        raise ValueError(f"Expected feature shape [C,H,W], got {feature_shape}")
    model = ResidualFusedFeatureAdapter(
        channels=feature_shape[0],
        hidden_channels=int(args.hidden_channels),
        blocks=int(args.blocks),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    metadata = {
        "mode": "transfuserpp_fused_feature_adapter",
        "source_cache": str(Path(args.source_cache).expanduser()),
        "target_cache": str(Path(args.target_cache).expanduser()),
        "source_metadata": source_cache.metadata,
        "target_metadata": target_cache.metadata,
        "feature_shape": list(feature_shape),
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "args": vars(args),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"train_samples": len(train_ds), "val_samples": len(val_ds), "feature_shape": feature_shape}, indent=2), flush=True)
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_metrics = _run_epoch(model, val_loader, optimizer, device, args, train=False)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        val_loss = float(val_metrics["loss"])
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.6f} "
            f"val={val_loss:.6f} best={best_val if best_epoch else val_loss:.6f} "
            f"input_l1={val_metrics['input_l1']:.6f} adapted_l1={val_metrics['adapted_l1']:.6f}",
            flush=True,
        )
        if val_loss + float(args.early_stop_min_delta) < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
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
    summary = {"best_epoch": int(best_epoch), "best_val_loss": float(best_val), "mode": "transfuserpp_fused_feature_adapter"}
    if (out_dir / "best_model.pt").exists():
        best = torch.load(out_dir / "best_model.pt", map_location="cpu")
        summary.update(best.get("val_metrics", {}))
    print(json.dumps(summary, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a residual adapter on TransFuser++ fused backbone features.")
    parser.add_argument("--source-cache", required=True, help="Shifted feature cache directory.")
    parser.add_argument("--target-cache", required=True, help="Canonical feature cache directory.")
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
    parser.add_argument("--feature-loss-weight", type=float, default=1.0)
    parser.add_argument("--cosine-loss-weight", type=float, default=0.05)
    parser.add_argument("--residual-loss-weight", type=float, default=0.01)
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
