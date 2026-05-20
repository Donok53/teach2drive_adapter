import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .cache_transfuserpp_feature_fusion_features import FUSED_FEATURE_NAME, STAGE_FEATURE_NAMES
from .train_transfuserpp_fused_feature_adapter import ResidualFusedFeatureAdapter, _feature_cosine_loss, _split_by_episode


def _load_metadata(cache_dir: Path) -> Dict:
    path = cache_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature-fusion cache metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class FeatureFusionCache:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.metadata = _load_metadata(self.cache_dir)
        self.stage_names = tuple(self.metadata.get("stage_feature_names", STAGE_FEATURE_NAMES))
        self.fused_name = str(self.metadata.get("fused_feature_name", FUSED_FEATURE_NAME))
        self.feature_names = (*self.stage_names, self.fused_name)
        self.features = {
            name: np.load(self.cache_dir / f"{name}.npy", mmap_mode="r")
            for name in self.feature_names
        }
        self.sample_episode = np.load(self.cache_dir / "sample_episode.npy", mmap_mode="r")
        self.sample_frame = np.load(self.cache_dir / "sample_frame.npy", mmap_mode="r")
        self.sample_index = np.load(self.cache_dir / "sample_index.npy", mmap_mode="r")
        for name, features in self.features.items():
            if len(features) != len(self.sample_episode):
                raise ValueError(f"cache length mismatch for {name} in {self.cache_dir}")


class FeatureFusionPairDataset(Dataset):
    def __init__(self, source: FeatureFusionCache, target: FeatureFusionCache, indices: Optional[np.ndarray] = None) -> None:
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

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        source = {
            name: torch.from_numpy(np.array(self.source.features[name][idx], copy=True))
            for name in self.source.feature_names
        }
        target = {
            name: torch.from_numpy(np.array(self.target.features[name][idx], copy=True))
            for name in self.target.feature_names
        }
        return {
            "source": source,
            "target": target,
            "episode": torch.tensor(int(self.source.sample_episode[idx]), dtype=torch.long),
            "frame": torch.tensor(int(self.source.sample_frame[idx]), dtype=torch.long),
        }


class FeatureThenFusionAdapter(nn.Module):
    """Adapters at fusion-stage token maps, followed by a final fused-feature adapter."""

    def __init__(
        self,
        stage_feature_shapes: Dict[str, tuple[int, int, int]],
        fused_feature_shape: tuple[int, int, int],
        hidden_channels: int = 0,
        blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.stage_names = tuple(stage_feature_shapes.keys())
        self.stage_adapters = nn.ModuleDict(
            {
                name: ResidualFusedFeatureAdapter(
                    channels=int(shape[0]),
                    hidden_channels=int(hidden_channels),
                    blocks=int(blocks),
                    dropout=float(dropout),
                )
                for name, shape in stage_feature_shapes.items()
            }
        )
        self.fused_adapter = ResidualFusedFeatureAdapter(
            channels=int(fused_feature_shape[0]),
            hidden_channels=int(hidden_channels),
            blocks=int(blocks),
            dropout=float(dropout),
        )

    def adapt_stage(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.stage_adapters[name](x)

    def adapt_layer(self, layer_idx: int, image_features: torch.Tensor, lidar_features: torch.Tensor):
        image_name = f"layer_{int(layer_idx)}_image"
        lidar_name = f"layer_{int(layer_idx)}_lidar"
        return self.stage_adapters[image_name](image_features), self.stage_adapters[lidar_name](lidar_features)

    def adapt_fused(self, fused_features: torch.Tensor) -> torch.Tensor:
        return self.fused_adapter(fused_features)

    def forward(self, source: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {name: self.stage_adapters[name](source[name]) for name in self.stage_names}
        output[FUSED_FEATURE_NAME] = self.fused_adapter(source[FUSED_FEATURE_NAME])
        return output


def _move_feature_dict(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True).float() for name, value in batch.items()}


def _map_losses(pred: torch.Tensor, source: torch.Tensor, target: torch.Tensor, feature_weight: float, cosine_weight: float, residual_weight: float):
    feature_loss = nn.functional.smooth_l1_loss(pred, target)
    cosine_loss = _feature_cosine_loss(pred, target)
    residual_loss = torch.mean(torch.abs(pred - source))
    loss = feature_weight * feature_loss + cosine_weight * cosine_loss + residual_weight * residual_loss
    input_l1 = torch.mean(torch.abs(source - target))
    adapted_l1 = torch.mean(torch.abs(pred.detach() - target))
    return loss, feature_loss, cosine_loss, residual_loss, input_l1, adapted_l1


def _run_epoch(model, loader, optimizer, device, args, train: bool) -> Dict[str, float]:
    model.train(train)
    metric_names = (
        "loss",
        "stage_feature",
        "stage_cosine",
        "stage_residual",
        "stage_input_l1",
        "stage_adapted_l1",
        "fused_feature",
        "fused_cosine",
        "fused_residual",
        "fused_input_l1",
        "fused_adapted_l1",
    )
    totals = {name: 0.0 for name in metric_names}
    totals["samples"] = 0
    start = time.time()
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    stage_names = tuple(raw_model.stage_names)

    for step, batch in enumerate(loader, start=1):
        source = _move_feature_dict(batch["source"], device)
        target = _move_feature_dict(batch["target"], device)
        with torch.set_grad_enabled(train):
            pred = model(source)
            stage_losses = []
            stage_feature = []
            stage_cosine = []
            stage_residual = []
            stage_input_l1 = []
            stage_adapted_l1 = []
            for name in stage_names:
                values = _map_losses(
                    pred[name],
                    source[name],
                    target[name],
                    float(args.stage_feature_loss_weight),
                    float(args.stage_cosine_loss_weight),
                    float(args.stage_residual_loss_weight),
                )
                stage_losses.append(values[0])
                stage_feature.append(values[1])
                stage_cosine.append(values[2])
                stage_residual.append(values[3])
                stage_input_l1.append(values[4])
                stage_adapted_l1.append(values[5])
            stage_loss = torch.stack(stage_losses).mean()
            fused_values = _map_losses(
                pred[FUSED_FEATURE_NAME],
                source[FUSED_FEATURE_NAME],
                target[FUSED_FEATURE_NAME],
                float(args.fused_feature_loss_weight),
                float(args.fused_cosine_loss_weight),
                float(args.fused_residual_loss_weight),
            )
            loss = float(args.stage_loss_weight) * stage_loss + float(args.fused_loss_weight) * fused_values[0]
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if float(args.grad_clip) > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                optimizer.step()

        batch_size = int(source[FUSED_FEATURE_NAME].shape[0])
        stage_feature_value = torch.stack(stage_feature).mean()
        stage_cosine_value = torch.stack(stage_cosine).mean()
        stage_residual_value = torch.stack(stage_residual).mean()
        stage_input_l1_value = torch.stack(stage_input_l1).mean()
        stage_adapted_l1_value = torch.stack(stage_adapted_l1).mean()
        values = {
            "loss": loss,
            "stage_feature": stage_feature_value,
            "stage_cosine": stage_cosine_value,
            "stage_residual": stage_residual_value,
            "stage_input_l1": stage_input_l1_value,
            "stage_adapted_l1": stage_adapted_l1_value,
            "fused_feature": fused_values[1],
            "fused_cosine": fused_values[2],
            "fused_residual": fused_values[3],
            "fused_input_l1": fused_values[4],
            "fused_adapted_l1": fused_values[5],
        }
        for name, value in values.items():
            totals[name] += float(value.detach().cpu()) * batch_size
        totals["samples"] += batch_size
        if train and int(args.step_log_every) > 0 and (step == 1 or step % int(args.step_log_every) == 0):
            samples = max(totals["samples"], 1)
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"step={step:05d}/{len(loader):05d} "
                f"loss={totals['loss']/samples:.6f} "
                f"stage={totals['stage_feature']/samples:.6f} "
                f"stage_l1={totals['stage_adapted_l1']/samples:.6f} "
                f"fused={totals['fused_feature']/samples:.6f} "
                f"fused_l1={totals['fused_adapted_l1']/samples:.6f} "
                f"samples/s={samples/elapsed:.1f}",
                flush=True,
            )
    samples = max(int(totals.pop("samples")), 1)
    return {key: value / samples for key, value in totals.items()}


def _shape_dict(cache: FeatureFusionCache) -> Dict[str, tuple[int, int, int]]:
    return {name: tuple(int(v) for v in cache.features[name].shape[1:]) for name in cache.stage_names}


def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_cache = FeatureFusionCache(args.source_cache)
    target_cache = FeatureFusionCache(args.target_cache)
    train_indices, val_indices = _split_by_episode(source_cache.sample_episode, float(args.val_ratio), int(args.seed))
    if int(args.max_train_samples) > 0:
        train_indices = train_indices[: int(args.max_train_samples)]
    if int(args.max_val_samples) > 0:
        val_indices = val_indices[: int(args.max_val_samples)]
    train_ds = FeatureFusionPairDataset(source_cache, target_cache, train_indices)
    val_ds = FeatureFusionPairDataset(source_cache, target_cache, val_indices)
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
    model = FeatureThenFusionAdapter(
        stage_feature_shapes=stage_feature_shapes,
        fused_feature_shape=fused_feature_shape,
        hidden_channels=int(args.hidden_channels),
        blocks=int(args.blocks),
        dropout=float(args.dropout),
    ).to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    metadata = {
        "mode": "transfuserpp_feature_then_fusion_adapter",
        "source_cache": str(Path(args.source_cache).expanduser()),
        "target_cache": str(Path(args.target_cache).expanduser()),
        "source_metadata": source_cache.metadata,
        "target_metadata": target_cache.metadata,
        "stage_feature_shapes": {key: list(value) for key, value in stage_feature_shapes.items()},
        "fused_feature_shape": list(fused_feature_shape),
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
            },
            indent=2,
        ),
        flush=True,
    )
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_metrics = _run_epoch(model, val_loader, optimizer, device, args, train=False)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        val_loss = float(val_metrics["loss"])
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.6f} "
            f"val={val_loss:.6f} best={best_val if best_epoch else val_loss:.6f} "
            f"stage_l1={val_metrics['stage_adapted_l1']:.6f} "
            f"fused_l1={val_metrics['fused_adapted_l1']:.6f}",
            flush=True,
        )
        if val_loss + float(args.early_stop_min_delta) < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "model_state": raw_model.state_dict(),
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
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    summary = {"best_epoch": int(best_epoch), "best_val_loss": float(best_val), "mode": "transfuserpp_feature_then_fusion_adapter"}
    if (out_dir / "best_model.pt").exists():
        best = torch.load(out_dir / "best_model.pt", map_location="cpu")
        summary.update(best.get("val_metrics", {}))
    print(json.dumps(summary, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train feature-stage adapters followed by a TransFuser++ fused-feature adapter.")
    parser.add_argument("--source-cache", required=True, help="Shifted feature-fusion cache directory.")
    parser.add_argument("--target-cache", required=True, help="Canonical feature-fusion cache directory.")
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
    parser.add_argument("--stage-loss-weight", type=float, default=1.0)
    parser.add_argument("--fused-loss-weight", type=float, default=1.0)
    parser.add_argument("--stage-feature-loss-weight", type=float, default=1.0)
    parser.add_argument("--stage-cosine-loss-weight", type=float, default=0.05)
    parser.add_argument("--stage-residual-loss-weight", type=float, default=0.01)
    parser.add_argument("--fused-feature-loss-weight", type=float, default=1.0)
    parser.add_argument("--fused-cosine-loss-weight", type=float, default=0.05)
    parser.add_argument("--fused-residual-loss-weight", type=float, default=0.01)
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
