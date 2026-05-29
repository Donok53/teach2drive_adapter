#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Smoke run for the expert-only residual policy adapter on the partially
# collected Town13 Tesla dataset.
#
# This is not the paired feature/fusion alignment experiment. The supervision
# target is the expert trajectory/speed labels from the target vehicle/sensor
# dataset; frozen TransFuser++ predictions are used as a residual base/prior.

export SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_town13_paired_tfpp_ego_front_triplet_3h"}
export PY=${PY:-python}
export SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_town13_paired_tfpp_ego_front_triplet_smoke_complete"}
export SNAPSHOT_COMPLETE_EPISODES=${SNAPSHOT_COMPLETE_EPISODES:-1}
export REFRESH_SNAPSHOT=${REFRESH_SNAPSHOT:-1}
export SNAPSHOT_MIN_FRAMES=${SNAPSHOT_MIN_FRAMES:-1000}
export SNAPSHOT_REQUIRED_PROFILES=${SNAPSHOT_REQUIRED_PROFILES:-"front_triplet_shifted"}

if [[ "$SNAPSHOT_COMPLETE_EPISODES" == "1" ]]; then
  echo "=== snapshot complete episodes from $SOURCE_DATA_ROOT -> $SNAPSHOT_ROOT"
  SOURCE_DATA_ROOT="$SOURCE_DATA_ROOT" \
  SNAPSHOT_ROOT="$SNAPSHOT_ROOT" \
  REFRESH_SNAPSHOT="$REFRESH_SNAPSHOT" \
  SNAPSHOT_MIN_FRAMES="$SNAPSHOT_MIN_FRAMES" \
  SNAPSHOT_REQUIRED_PROFILES="$SNAPSHOT_REQUIRED_PROFILES" \
  "$PY" - <<'PY'
import json
import os
import shutil
from pathlib import Path

src = Path(os.environ["SOURCE_DATA_ROOT"]).expanduser()
dst = Path(os.environ["SNAPSHOT_ROOT"]).expanduser()
min_frames = int(os.environ.get("SNAPSHOT_MIN_FRAMES", "1000"))
profiles = [p.strip() for p in os.environ.get("SNAPSHOT_REQUIRED_PROFILES", "").split(",") if p.strip()]

if os.environ.get("REFRESH_SNAPSHOT", "1") == "1" and dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)

accepted = []
rejected = []
for ep in sorted(src.glob("episode_*")):
    if not ep.is_dir():
        continue
    frames = ep / "frames.jsonl"
    if not frames.exists():
        rejected.append((ep.name, "missing frames.jsonl"))
        continue
    missing_layouts = [
        profile
        for profile in profiles
        if not (ep / "rigs" / profile / "sensor_layout.json").exists()
    ]
    if missing_layouts:
        rejected.append((ep.name, "missing layouts=" + ",".join(missing_layouts)))
        continue
    try:
        frame_count = sum(1 for _ in frames.open("r", encoding="utf-8"))
    except OSError as exc:
        rejected.append((ep.name, f"frames read failed: {exc}"))
        continue
    if frame_count < min_frames:
        rejected.append((ep.name, f"frames={frame_count} < {min_frames}"))
        continue
    target = dst / ep.name
    if not target.exists():
        target.symlink_to(ep, target_is_directory=True)
    accepted.append((ep.name, frame_count))

print(json.dumps({
    "source": str(src),
    "snapshot": str(dst),
    "accepted": len(accepted),
    "accepted_tail": accepted[-8:],
    "rejected": len(rejected),
    "rejected_tail": rejected[-8:],
}, indent=2, ensure_ascii=False))
if not accepted:
    raise SystemExit("no complete episodes available for training")
PY
  export DATA_ROOT="$SNAPSHOT_ROOT"
else
  export DATA_ROOT=${DATA_ROOT:-"$SOURCE_DATA_ROOT"}
fi

export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_tesla_town13_residual_policy_adapter_smoke"}
export OUT=${OUT:-"$WORK_ROOT/train_front_triplet_shifted_residual_policy_adapter_smoke"}

# Train on the target sensor rig. Override PROFILE=tfpp_ego if you want to
# train/evaluate the original TransFuser++ camera layout instead.
export PROFILE=${PROFILE:-front_triplet_shifted}
export CAMERAS=${CAMERAS:-left,front,right}

# Keep this first run short and conservative because the dataset is still small.
export EPOCHS=${EPOCHS:-18}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-6}
export BATCH_SIZE=${BATCH_SIZE:-128}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
export NUM_WORKERS=${NUM_WORKERS:-8}
export CACHE_WORKERS=${CACHE_WORKERS:-16}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export CACHE_DATA_PARALLEL=${CACHE_DATA_PARALLEL:-$DATA_PARALLEL}
export TRAIN_DATA_PARALLEL=${TRAIN_DATA_PARALLEL:-$DATA_PARALLEL}
export OVERWRITE=${OVERWRITE:-0}

# Pure residual policy adapter: no paired canonical feature/fusion target.
export FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT=${FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT:-""}
export INIT_CHECKPOINT=${INIT_CHECKPOINT:-""}

export HIDDEN_DIM=${HIDDEN_DIM:-512}
export LAYOUT_HIDDEN_DIM=${LAYOUT_HIDDEN_DIM:-128}
export VISUAL_DIM=${VISUAL_DIM:-256}
export VISUAL_TOKEN_DIM=${VISUAL_TOKEN_DIM:-192}
export VISUAL_LAYERS=${VISUAL_LAYERS:-2}
export VISUAL_HEADS=${VISUAL_HEADS:-4}
export LR=${LR:-5e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-2e-4}

# Expert labels dominate, but the residual stays close to frozen TransFuser++
# at the beginning to reduce small-data collapse.
export TEACHER_TARGET_BLEND=${TEACHER_TARGET_BLEND:-0.0}
export TEACHER_TRAJ_BLEND=${TEACHER_TRAJ_BLEND:-0.0}
export TEACHER_SPEED_TARGET_BLEND=${TEACHER_SPEED_TARGET_BLEND:-0.0}
export TEACHER_STOP_TARGET_BLEND=${TEACHER_STOP_TARGET_BLEND:-0.0}
export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.10}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.05}
export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.25}

export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.10}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-1.00}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-1.50}

export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-0.45}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.02}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.75}
export SPEED_FLOOR_MASK=${SPEED_FLOOR_MASK:-target}
export SPEED_FLOOR_TARGET_THRESHOLD=${SPEED_FLOOR_TARGET_THRESHOLD:-2.0}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.03}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-0.8}

export STOP_SPEED_TARGET_THRESHOLD=${STOP_SPEED_TARGET_THRESHOLD:-0.5}
export STOP_SPEED_CEILING_MPS=${STOP_SPEED_CEILING_MPS:-0.6}
export STOP_SPEED_CEILING_LOSS_WEIGHT=${STOP_SPEED_CEILING_LOSS_WEIGHT:-0.25}
export HAZARD_STOP_SPEED_CEILING_LOSS_WEIGHT=${HAZARD_STOP_SPEED_CEILING_LOSS_WEIGHT:-0.35}

export LAUNCH_SAMPLE_WEIGHT=${LAUNCH_SAMPLE_WEIGHT:-1.25}
export LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=${LAUNCH_SPEED_FLOOR_LOSS_WEIGHT:-0.04}
export LAUNCH_SPEED_FLOOR_MPS=${LAUNCH_SPEED_FLOOR_MPS:-1.2}
export RELEASE_SAMPLE_WEIGHT=${RELEASE_SAMPLE_WEIGHT:-1.25}
export RELEASE_SPEED_FLOOR_LOSS_WEIGHT=${RELEASE_SPEED_FLOOR_LOSS_WEIGHT:-0.04}
export RELEASE_SPEED_FLOOR_MPS=${RELEASE_SPEED_FLOOR_MPS:-1.2}

export SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.03}
export SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.01}
export TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.02}
export TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.01}

# First pass: deploy through PID, so do not train direct throttle/brake.
export CONTROL_LOSS_WEIGHT=${CONTROL_LOSS_WEIGHT:-0.0}
export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-9999}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.0}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.0}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.0}

exec bash configs/train_tfpp_vehicle_b_expert_v1_server.sh
