from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class LateralPIDSpec:
    kp: float = 3.118357247806046
    kd: float = 1.3782508892109167
    ki: float = 0.6406067986034124
    speed_scale: float = 0.9755321901954155
    speed_offset: float = 1.9152884533402488
    minimum_lookahead: float = 24.0
    maximum_lookahead: float = 105.0
    window_size: int = 6
    runtime_hz: float = 20.0
    dataset_hz: float = 4.0


def _pad_or_trim(x: torch.Tensor, width: int) -> torch.Tensor:
    if x.shape[1] == width:
        return x
    if x.shape[1] > width:
        return x[:, :width]
    return nn.functional.pad(x, (0, width - x.shape[1]))


class TurnGatedSteerResidualAdapter(nn.Module):
    """A bounded Tesla-specific correction applied after the frozen TF++ PID.

    The gate is deterministic and exactly zero on straight segments. This keeps
    the pretrained policy as the default and prevents a small 3-hour dataset
    from rewriting braking, throttle, perception, or route intent.
    """

    def __init__(
        self,
        checkpoint_dim: int = 20,
        hidden_dim: int = 128,
        max_delta: float = 0.20,
        turn_threshold: float = 0.035,
        full_turn_threshold: float = 0.12,
        dropout: float = 0.05,
        use_yaw_rate: bool = False,
        adapter_mode: str = "residual",
        minimum_gain: float = 0.10,
        maximum_gain: float = 1.20,
    ) -> None:
        super().__init__()
        self.checkpoint_dim = int(checkpoint_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_delta = float(max_delta)
        self.turn_threshold = float(turn_threshold)
        self.full_turn_threshold = float(full_turn_threshold)
        self.use_yaw_rate = bool(use_yaw_rate)
        self.adapter_mode = str(adapter_mode)
        self.minimum_gain = float(minimum_gain)
        self.maximum_gain = float(maximum_gain)
        if self.adapter_mode not in {"residual", "gain"}:
            raise ValueError(f"adapter_mode must be residual or gain, got {self.adapter_mode!r}")
        if not self.minimum_gain < 1.0 < self.maximum_gain:
            raise ValueError("minimum_gain < 1 < maximum_gain is required")
        input_dim = self.checkpoint_dim + 7 + int(self.use_yaw_rate)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        if self.adapter_mode == "gain":
            identity_fraction = (1.0 - self.minimum_gain) / (self.maximum_gain - self.minimum_gain)
            identity_logit = math.log(identity_fraction / (1.0 - identity_fraction))
            nn.init.constant_(self.net[-1].bias, identity_logit)
        else:
            nn.init.zeros_(self.net[-1].bias)

    def turn_gate(
        self,
        checkpoint_flat: torch.Tensor,
        base_steer: torch.Tensor,
        pid_error: torch.Tensor,
    ) -> torch.Tensor:
        checkpoint = _pad_or_trim(checkpoint_flat, self.checkpoint_dim).reshape(-1, self.checkpoint_dim // 2, 2)
        final_heading = torch.atan2(checkpoint[:, -1, 1], checkpoint[:, -1, 0].clamp_min(1e-3)).abs()
        # PID error is normalized by pi/2 in CARLA Garage. Base steer provides a
        # second signal when derivative/integral terms identify turn entry/exit.
        strength = torch.maximum(pid_error.abs(), 0.35 * base_steer.abs())
        strength = torch.maximum(strength, final_heading / (0.5 * math.pi))
        width = max(self.full_turn_threshold - self.turn_threshold, 1e-6)
        return ((strength - self.turn_threshold) / width).clamp(0.0, 1.0)

    def forward(
        self,
        checkpoint_flat: torch.Tensor,
        current_speed: torch.Tensor,
        target_speed: torch.Tensor,
        base_steer: torch.Tensor,
        pid_error: torch.Tensor,
        pid_derivative: torch.Tensor,
        pid_integral: torch.Tensor,
        yaw_rate: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        checkpoint_flat = _pad_or_trim(checkpoint_flat.float(), self.checkpoint_dim)
        vectors = [
            checkpoint_flat / 10.0,
            current_speed.reshape(-1, 1).float() / 20.0,
            target_speed.reshape(-1, 1).float() / 20.0,
            base_steer.reshape(-1, 1).float(),
            pid_error.reshape(-1, 1).float(),
            pid_derivative.reshape(-1, 1).float(),
            pid_integral.reshape(-1, 1).float(),
            (current_speed.reshape(-1, 1).float() * pid_error.reshape(-1, 1).float()) / 20.0,
        ]
        if self.use_yaw_rate:
            if yaw_rate is None:
                yaw_rate = torch.zeros_like(current_speed)
            vectors.append(yaw_rate.reshape(-1, 1).float().clamp(-2.0, 2.0))
        raw_value = self.net(torch.cat(vectors, dim=1))
        gate = self.turn_gate(checkpoint_flat, base_steer, pid_error).reshape(-1, 1)
        base = base_steer.reshape(-1, 1)
        if self.adapter_mode == "gain":
            raw_gain = self.minimum_gain + (self.maximum_gain - self.minimum_gain) * torch.sigmoid(raw_value)
            gain = 1.0 + gate * (raw_gain - 1.0)
            delta = base * (gain - 1.0)
            raw_delta = base * (raw_gain - 1.0)
        else:
            raw_delta = torch.tanh(raw_value) * self.max_delta
            delta = raw_delta * gate
            raw_gain = torch.ones_like(raw_delta)
            gain = torch.ones_like(raw_delta)
        steer = (base + delta).clamp(-1.0, 1.0)
        return {
            "steer": steer, "delta": delta, "raw_delta": raw_delta,
            "gate": gate, "gain": gain, "raw_gain": raw_gain,
        }


def _split_by_episode(sample_episode: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(sample_episode)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * float(val_ratio))))
    val_episodes = set(int(v) for v in episodes[:val_count])
    val_mask = np.asarray([int(v) in val_episodes for v in sample_episode], dtype=bool)
    indices = np.arange(len(sample_episode), dtype=np.int64)
    return indices[~val_mask], indices[val_mask]


def _pid_inputs(
    checkpoint_flat: np.ndarray,
    scalar: np.ndarray,
    sample_episode: np.ndarray,
    sample_frame: np.ndarray,
    spec: LateralPIDSpec,
) -> Dict[str, np.ndarray]:
    count = len(checkpoint_flat)
    base = np.zeros(count, dtype=np.float32)
    error_out = np.zeros(count, dtype=np.float32)
    derivative_out = np.zeros(count, dtype=np.float32)
    integral_out = np.zeros(count, dtype=np.float32)
    checkpoints = checkpoint_flat.reshape(count, -1, 2)
    substeps = max(1, int(round(spec.runtime_hz / spec.dataset_hz)))

    for episode in np.unique(sample_episode):
        ids = np.flatnonzero(sample_episode == episode)
        ids = ids[np.argsort(sample_frame[ids])]
        history: list[float] = []
        previous_error: Optional[float] = None
        for idx in ids:
            speed = abs(float(scalar[idx, 0]))
            lookahead = spec.speed_scale * speed * 3.6 + spec.speed_offset
            lookahead = float(np.clip(lookahead, spec.minimum_lookahead, spec.maximum_lookahead))
            point_idx = min(int(lookahead), checkpoints.shape[1] - 1)
            point = checkpoints[idx, point_idx]
            error = float(math.atan2(float(point[1]), max(float(point[0]), 1e-3)) / (0.5 * math.pi))
            start = error if previous_error is None else previous_error
            for substep in range(1, substeps + 1):
                value = start + (error - start) * (substep / substeps)
                history.append(value)
                history = history[-int(spec.window_size) :]
            derivative = 0.0 if len(history) < 2 else history[-1] - history[-2]
            integral = float(np.mean(history)) if history else 0.0
            base[idx] = float(np.clip(spec.kp * error + spec.kd * derivative + spec.ki * integral, -1.0, 1.0))
            error_out[idx] = error
            derivative_out[idx] = derivative
            integral_out[idx] = integral
            previous_error = error
    return {
        "base_steer": base,
        "pid_error": error_out,
        "pid_derivative": derivative_out,
        "pid_integral": integral_out,
    }


def _derive_yaw_rate(
    cache_metadata: Dict,
    sample_episode: np.ndarray,
    sample_frame: np.ndarray,
    episode_root: str,
) -> np.ndarray:
    """Recover current ego yaw rate from consecutive recorded odometry poses."""
    root = Path(episode_root).expanduser()
    index_path = Path(str(cache_metadata.get("index", ""))).expanduser()
    episode_dirs = None
    if index_path.is_file():
        index_arrays = np.load(index_path, allow_pickle=True)
        if "episode_dirs" in index_arrays.files:
            raw_dirs = [Path(str(value)) for value in index_arrays["episode_dirs"]]
            episode_dirs = [root / path.name for path in raw_dirs]

    yaw_rate = np.zeros(len(sample_episode), dtype=np.float32)
    for episode in np.unique(sample_episode):
        episode_id = int(episode)
        episode_dir = (
            episode_dirs[episode_id]
            if episode_dirs is not None and episode_id < len(episode_dirs)
            else root / f"episode_{episode_id:06d}"
        )
        frames_path = episode_dir / "frames.jsonl"
        if not frames_path.is_file():
            raise FileNotFoundError(f"Cannot derive yaw rate: missing {frames_path}")
        frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = np.flatnonzero(sample_episode == episode_id)
        for idx in ids:
            frame_idx = int(sample_frame[idx])
            if frame_idx <= 0 or frame_idx >= len(frames):
                continue
            current = frames[frame_idx]
            previous = frames[frame_idx - 1]
            dt = float(current.get("time", 0.0)) - float(previous.get("time", 0.0))
            if dt <= 1e-4:
                continue
            yaw = float(current.get("odom", {}).get("yaw", 0.0))
            previous_yaw = float(previous.get("odom", {}).get("yaw", yaw))
            delta = math.atan2(math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw))
            yaw_rate[idx] = np.float32(delta / dt)
    return yaw_rate


class CachedSteerDataset(Dataset):
    def __init__(
        self,
        cache_path: str,
        indices: Optional[np.ndarray] = None,
        pid_spec: Optional[LateralPIDSpec] = None,
        episode_root: str = "",
        yaw_rate: Optional[np.ndarray] = None,
    ) -> None:
        arrays = np.load(Path(cache_path).expanduser(), allow_pickle=True)
        required = {"control_target", "control_mask", "checkpoint_flat", "expected_speed"}
        missing = sorted(required.difference(arrays.files))
        if missing:
            raise KeyError(f"Cache is missing vehicle-control arrays: {missing}")
        self.scalar = arrays["scalar"].astype(np.float32)
        self.target = arrays["target"].astype(np.float32)
        self.control = arrays["control_target"].astype(np.float32)
        self.control_mask = arrays["control_mask"].astype(np.float32).reshape(-1)
        self.checkpoint = arrays["checkpoint_flat"].astype(np.float32)
        self.target_speed = arrays["expected_speed"].astype(np.float32).reshape(-1)
        self.sample_episode = arrays["sample_episode"].astype(np.int64)
        self.sample_frame = arrays["sample_frame"].astype(np.int64)
        self.metadata = json.loads(str(arrays["metadata"].item())) if "metadata" in arrays.files else {}
        self.pid_spec = pid_spec or LateralPIDSpec()
        self.pid = _pid_inputs(self.checkpoint, self.scalar, self.sample_episode, self.sample_frame, self.pid_spec)
        if yaw_rate is not None:
            self.yaw_rate = yaw_rate.astype(np.float32, copy=False)
        elif episode_root:
            self.yaw_rate = _derive_yaw_rate(
                self.metadata, self.sample_episode, self.sample_frame, episode_root
            )
        else:
            self.yaw_rate = np.zeros(len(self.scalar), dtype=np.float32)
        self.indices = np.arange(len(self.scalar), dtype=np.int64) if indices is None else indices.astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        return {
            "checkpoint": torch.from_numpy(self.checkpoint[idx]),
            "current_speed": torch.tensor(abs(float(self.scalar[idx, 0])), dtype=torch.float32),
            "target_speed": torch.tensor(float(self.target_speed[idx]), dtype=torch.float32),
            "base_steer": torch.tensor(float(self.pid["base_steer"][idx]), dtype=torch.float32),
            "pid_error": torch.tensor(float(self.pid["pid_error"][idx]), dtype=torch.float32),
            "pid_derivative": torch.tensor(float(self.pid["pid_derivative"][idx]), dtype=torch.float32),
            "pid_integral": torch.tensor(float(self.pid["pid_integral"][idx]), dtype=torch.float32),
            "yaw_rate": torch.tensor(float(self.yaw_rate[idx]), dtype=torch.float32),
            "expert_steer": torch.tensor(float(self.control[idx, 0]), dtype=torch.float32),
            "expert_brake": torch.tensor(float(self.control[idx, 2]), dtype=torch.float32),
            "control_mask": torch.tensor(float(self.control_mask[idx]), dtype=torch.float32),
            "future_traj": torch.from_numpy(self.target[idx, :12].reshape(4, 3)),
        }


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(value * weight) / torch.clamp(torch.sum(weight), min=1e-6)


def _run_epoch(model, loader, optimizer, device, args, train: bool) -> Dict[str, float]:
    model.train(train)
    totals = {k: 0.0 for k in (
        "loss", "base_mae", "adapted_mae", "turn_base_mae", "turn_adapted_mae",
        "straight_delta", "residual_abs", "residual_signed", "gate", "turn_ratio",
        "straight_ratio", "usable_ratio", "target_clip_ratio", "yaw_rate_abs",
        "gain", "target_gain",
    )}
    samples = 0
    for batch in loader:
        tensors = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
        with torch.set_grad_enabled(train):
            out = model(
                tensors["checkpoint"], tensors["current_speed"], tensors["target_speed"],
                tensors["base_steer"], tensors["pid_error"], tensors["pid_derivative"], tensors["pid_integral"],
                tensors["yaw_rate"],
            )
            expert = tensors["expert_steer"].reshape(-1, 1).clamp(-1.0, 1.0)
            active = (
                (tensors["control_mask"].reshape(-1, 1) > 0.5)
                & (tensors["current_speed"].reshape(-1, 1) >= float(args.minimum_speed))
                & (tensors["expert_brake"].reshape(-1, 1) < 0.5)
            ).float()
            gate = out["gate"].detach()
            turn = (gate >= 0.5).float()
            straight = (gate <= float(args.straight_gate_threshold)).float()
            base = tensors["base_steer"].reshape(-1, 1)
            direction_consistent = (
                (base * expert >= 0.0)
                | (base.abs() <= float(args.opposite_steer_deadband))
                | (expert.abs() <= float(args.opposite_steer_deadband))
            ).float()
            if str(args.adapter_mode) == "gain":
                reliable_magnitude = (
                    (base.abs() >= float(args.minimum_base_steer))
                    & (expert.abs() >= float(args.minimum_expert_steer))
                ).float()
            else:
                reliable_magnitude = torch.ones_like(active)
            usable = active * direction_consistent * reliable_magnitude
            action_weight = usable * gate.clamp_min(float(args.minimum_turn_loss_gate))
            requested_delta = expert - base
            target_delta = requested_delta.clamp(-float(args.max_delta), float(args.max_delta))
            requested_gain = (expert.abs() / base.abs().clamp_min(1e-3))
            target_gain = requested_gain.clamp(float(args.minimum_gain), float(args.maximum_gain))
            if str(args.adapter_mode) == "gain":
                action_raw = nn.functional.smooth_l1_loss(
                    out["raw_gain"], target_gain, reduction="none", beta=0.05
                )
                clipped = (requested_gain < float(args.minimum_gain)) | (requested_gain > float(args.maximum_gain))
            else:
                action_raw = nn.functional.smooth_l1_loss(
                    out["delta"], target_delta, reduction="none", beta=0.03
                )
                clipped = requested_delta.abs() > float(args.max_delta)
            action_loss = _weighted_mean(action_raw, action_weight)
            straight_weight = active * straight
            identity_loss = _weighted_mean(out["delta"].square(), straight_weight)
            magnitude_loss = _weighted_mean(out["delta"].square(), active)
            loss = action_loss + float(args.straight_identity_weight) * identity_loss + float(args.residual_magnitude_weight) * magnitude_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

        batch_size = int(expert.shape[0])
        base_abs = (tensors["base_steer"].reshape(-1, 1) - expert).abs()
        adapted_abs = (out["steer"] - expert).abs()
        turn_active = active * turn * direction_consistent
        values = {
            "loss": loss.detach(),
            "base_mae": _weighted_mean(base_abs, active),
            "adapted_mae": _weighted_mean(adapted_abs, active),
            "turn_base_mae": _weighted_mean(base_abs, turn_active),
            "turn_adapted_mae": _weighted_mean(adapted_abs, turn_active),
            "straight_delta": _weighted_mean(out["delta"].abs(), straight_weight),
            "residual_abs": _weighted_mean(out["delta"].abs(), active),
            "residual_signed": _weighted_mean(out["delta"], active),
            "gate": out["gate"].mean(),
            "turn_ratio": turn.mean(),
            "straight_ratio": straight.mean(),
            "usable_ratio": usable.mean(),
            "target_clip_ratio": _weighted_mean(clipped.float(), action_weight),
            "yaw_rate_abs": tensors["yaw_rate"].abs().mean(),
            "gain": _weighted_mean(out["gain"], action_weight),
            "target_gain": _weighted_mean(target_gain, action_weight),
        }
        for key, value in values.items():
            totals[key] += float(value.detach().cpu()) * batch_size
        samples += batch_size
    return {key: value / max(samples, 1) for key, value in totals.items()}


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    all_ds = CachedSteerDataset(args.cache, episode_root=args.episode_root)
    train_ids, val_ids = _split_by_episode(all_ds.sample_episode, args.val_ratio, args.seed)
    train_ds = CachedSteerDataset(args.cache, train_ids, all_ds.pid_spec, yaw_rate=all_ds.yaw_rate)
    val_ds = CachedSteerDataset(args.cache, val_ids, all_ds.pid_spec, yaw_rate=all_ds.yaw_rate)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = TurnGatedSteerResidualAdapter(
        checkpoint_dim=int(all_ds.checkpoint.shape[1]), hidden_dim=args.hidden_dim, max_delta=args.max_delta,
        turn_threshold=args.turn_threshold, full_turn_threshold=args.full_turn_threshold, dropout=args.dropout,
        use_yaw_rate=bool(args.use_yaw_rate),
        adapter_mode=args.adapter_mode, minimum_gain=args.minimum_gain, maximum_gain=args.maximum_gain,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metadata = {
        "mode": "tfpp_vehicle_turn_gated_steer_residual",
        "cache": str(Path(args.cache).expanduser()),
        "cache_metadata": all_ds.metadata,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "pid_spec": asdict(all_ds.pid_spec),
        "vehicle_steer_adapter": {
            "enabled": True,
            "checkpoint_dim": int(all_ds.checkpoint.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "max_delta": float(args.max_delta),
            "turn_threshold": float(args.turn_threshold),
            "full_turn_threshold": float(args.full_turn_threshold),
            "dropout": float(args.dropout),
            "use_yaw_rate": bool(args.use_yaw_rate),
            "adapter_mode": str(args.adapter_mode),
            "minimum_gain": float(args.minimum_gain),
            "maximum_gain": float(args.maximum_gain),
        },
        "args": vars(args),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)

    def save(path: Path, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], selection: float) -> None:
        torch.save({
            "epoch": epoch,
            "vehicle_steer_adapter_state": model.state_dict(),
            "metadata": metadata,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "selection_value": selection,
        }, path)

    history = []
    best = float("inf")
    stale = 0
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_metrics = _run_epoch(model, val_loader, None, device, args, train=False)
        selection = val_metrics["turn_adapted_mae"] + float(args.selection_straight_weight) * val_metrics["straight_delta"]
        improved = selection < best - float(args.early_stop_min_delta)
        if improved:
            best = selection
            stale = 0
            save(out_dir / "best_model.pt", epoch, train_metrics, val_metrics, selection)
        else:
            stale += 1
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "selection": selection})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f} "
            f"select={selection:.6f} best={best:.6f} new_best={int(improved)} "
            f"mae={val_metrics['base_mae']:.5f}->{val_metrics['adapted_mae']:.5f} "
            f"turn_mae={val_metrics['turn_base_mae']:.5f}->{val_metrics['turn_adapted_mae']:.5f} "
            f"delta_abs/signed={val_metrics['residual_abs']:.5f}/{val_metrics['residual_signed']:.5f} "
            f"gain={val_metrics['gain']:.3f}->{val_metrics['target_gain']:.3f} "
            f"straight_delta={val_metrics['straight_delta']:.5f} gate={val_metrics['gate']:.3f} "
            f"turn/straight/usable={val_metrics['turn_ratio']:.3f}/{val_metrics['straight_ratio']:.3f}/{val_metrics['usable_ratio']:.3f} "
            f"clip={val_metrics['target_clip_ratio']:.3f} yaw_rate={val_metrics['yaw_rate_abs']:.3f} "
            f"elapsed_min={(time.time()-started)/60.0:.1f}",
            flush=True,
        )
        if stale >= int(args.early_stop_patience):
            print(f"early_stop: no improvement for {stale} epochs", flush=True)
            break


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a turn-gated residual on the frozen TF++ lateral PID output.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--episode-root", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-delta", type=float, default=0.20)
    parser.add_argument("--adapter-mode", choices=("residual", "gain"), default="residual")
    parser.add_argument("--minimum-gain", type=float, default=0.10)
    parser.add_argument("--maximum-gain", type=float, default=1.20)
    parser.add_argument("--minimum-base-steer", type=float, default=0.05)
    parser.add_argument("--minimum-expert-steer", type=float, default=0.05)
    parser.add_argument("--turn-threshold", type=float, default=0.035)
    parser.add_argument("--full-turn-threshold", type=float, default=0.12)
    parser.add_argument("--turn-sample-weight", type=float, default=6.0)
    parser.add_argument("--expert-steer-turn-threshold", type=float, default=0.08)
    parser.add_argument("--future-lateral-turn-threshold", type=float, default=0.50)
    parser.add_argument("--future-yaw-turn-threshold", type=float, default=0.08)
    parser.add_argument("--minimum-speed", type=float, default=0.5)
    parser.add_argument("--straight-identity-weight", type=float, default=4.0)
    parser.add_argument("--residual-magnitude-weight", type=float, default=0.25)
    parser.add_argument("--minimum-turn-loss-gate", type=float, default=0.05)
    parser.add_argument("--straight-gate-threshold", type=float, default=0.05)
    parser.add_argument("--opposite-steer-deadband", type=float, default=0.05)
    parser.add_argument("--selection-straight-weight", type=float, default=1.0)
    parser.add_argument("--use-yaw-rate", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
