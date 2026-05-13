#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep"}
WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_shifted_adapter"}
BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-"$HOME/teach2drive/workspace/teach2drive_bootstrap"}
GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}
TFPP_CHECKPOINT=${TFPP_CHECKPOINT:-""}

AUGMENTATIONS=${AUGMENTATIONS:-0}
EPOCHS=${EPOCHS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
CACHE_WORKERS=${CACHE_WORKERS:-16}
LR=${LR:-2e-4}
VAL_RATIO=${VAL_RATIO:-0.15}
SEED=${SEED:-41}
OVERWRITE=${OVERWRITE:-0}
DATA_PARALLEL=${DATA_PARALLEL:-1}
SKIP_INVALID_MOTION=${SKIP_INVALID_MOTION:-1}
MOVING_SPEED_THRESHOLD=${MOVING_SPEED_THRESHOLD:-1.0}
MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.0}
STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-1.0}
TEACHER_TARGET_BLEND=${TEACHER_TARGET_BLEND:-0.0}
XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-1.0}
YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.05}
SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.10}
SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.0}
SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.0}
SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.0}
SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-1.0}
SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.0}
SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.0}
TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.0}
TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.0}
PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.0}
STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.05}
STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.10}
STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.02}
STOP_POSITIVE_LOSS_SCALE=${STOP_POSITIVE_LOSS_SCALE:-1.0}
STOP_NEGATIVE_LOSS_SCALE=${STOP_NEGATIVE_LOSS_SCALE:-1.0}
STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-1}

VIEW_ROOT=${VIEW_ROOT:-"$WORK_ROOT/profile_views"}
TFPP_VIEW=${TFPP_VIEW:-"$VIEW_ROOT/tfpp_ego"}
SHIFT_VIEW=${SHIFT_VIEW:-"$VIEW_ROOT/front_triplet_shifted"}
INDEX_DIR=${INDEX_DIR:-"$WORK_ROOT/indexes"}
TFPP_INDEX=${TFPP_INDEX:-"$INDEX_DIR/tfpp_ego_index.npz"}
SHIFT_INDEX=${SHIFT_INDEX:-"$INDEX_DIR/front_triplet_shifted_index.npz"}
CACHE=${CACHE:-"$WORK_ROOT/cache/tfpp_ego_prior_cache.npz"}
SHIFT_CACHE=${SHIFT_CACHE:-"$WORK_ROOT/cache/front_triplet_shifted_prior_cache.npz"}
TRAIN_CACHE=${TRAIN_CACHE:-"$CACHE"}
TEACHER_CACHE=${TEACHER_CACHE:-""}
BUILD_SHIFT_CACHE=${BUILD_SHIFT_CACHE:-0}
OUT=${OUT:-"$WORK_ROOT/train_shifted_visual_layout_adapter"}

mkdir -p "$WORK_ROOT" "$INDEX_DIR" "$(dirname "$CACHE")" "$(dirname "$SHIFT_CACHE")" "$OUT"

EXPORT_ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
  EXPORT_ARGS+=(--overwrite)
fi
if [[ "$SKIP_INVALID_MOTION" == "1" || "$SKIP_INVALID_MOTION" == "true" || "$SKIP_INVALID_MOTION" == "TRUE" ]]; then
  EXPORT_ARGS+=(--skip-invalid-motion)
fi

echo "=== export tfpp_ego profile view"
"$PY" -m teach2drive_adapter.export_paired_profile_view \
  --input-root "$DATA_ROOT" \
  --output-root "$TFPP_VIEW" \
  --profile tfpp_ego \
  --require-cameras front \
  "${EXPORT_ARGS[@]}"

echo "=== export front_triplet_shifted profile view"
"$PY" -m teach2drive_adapter.export_paired_profile_view \
  --input-root "$DATA_ROOT" \
  --output-root "$SHIFT_VIEW" \
  --profile front_triplet_shifted \
  --require-cameras left,front,right \
  "${EXPORT_ARGS[@]}"

build_index() {
  local input_root=$1
  local output=$2
  local cameras=$3
  if [[ -f "$output" && "$OVERWRITE" != "1" ]]; then
    echo "=== reuse index $output"
    return
  fi
  echo "=== build index $output cameras=$cameras"
  (
    cd "$BOOTSTRAP_ROOT"
    PYTHONPATH="$BOOTSTRAP_ROOT:${PYTHONPATH:-}" "$PY" -m teach2drive.token_dataset \
      --input-root "$input_root" \
      --output "$output" \
      --cameras "$cameras" \
      --augmentations "$AUGMENTATIONS" \
      --pseudo-label-name "__missing_pseudo_labels__.jsonl" \
      --seed "$SEED"
  )
}

build_index "$TFPP_VIEW" "$TFPP_INDEX" "front"
build_index "$SHIFT_VIEW" "$SHIFT_INDEX" "left,front,right"

echo "=== verify index alignment"
"$PY" - "$TFPP_INDEX" "$SHIFT_INDEX" <<'PY'
import sys
import numpy as np

tfpp = np.load(sys.argv[1], allow_pickle=True)
shift = np.load(sys.argv[2], allow_pickle=True)
for key in ("sample_episode_indices", "sample_frame_indices"):
    if not np.array_equal(tfpp[key], shift[key]):
        raise SystemExit(f"index mismatch: {key}")
print({"samples": int(len(tfpp["sample_frame_indices"])), "aligned": True})
PY

CACHE_ARGS=()
if [[ "$DATA_PARALLEL" == "1" ]]; then
  CACHE_ARGS+=(--data-parallel)
fi
if [[ -n "$TFPP_CHECKPOINT" ]]; then
  CACHE_ARGS+=(--checkpoint "$TFPP_CHECKPOINT")
fi

if [[ -f "$CACHE" && "$OVERWRITE" != "1" ]]; then
  echo "=== reuse prior cache $CACHE"
else
  echo "=== cache official tfpp_ego prior"
  "$PY" -m teach2drive_adapter.cache_transfuserpp_prior \
    --index "$TFPP_INDEX" \
    --output "$CACHE" \
    --garage-root "$GARAGE_ROOT" \
    --team-config "$TEAM_CONFIG" \
    --episode-root-override "$TFPP_VIEW" \
    --cameras front \
    --tfpp-camera front \
    --command-mode target_angle \
    --image-size 640 360 \
    --lidar-size 128 \
    --batch-size "$CACHE_BATCH_SIZE" \
    --num-workers "$CACHE_WORKERS" \
    "${CACHE_ARGS[@]}"
fi

if [[ "$BUILD_SHIFT_CACHE" == "1" || "$BUILD_SHIFT_CACHE" == "true" || "$BUILD_SHIFT_CACHE" == "TRUE" ]]; then
  if [[ -f "$SHIFT_CACHE" && "$OVERWRITE" != "1" ]]; then
    echo "=== reuse shifted prior cache $SHIFT_CACHE"
  else
    echo "=== cache shifted front_triplet prior"
    "$PY" -m teach2drive_adapter.cache_transfuserpp_prior \
      --index "$SHIFT_INDEX" \
      --output "$SHIFT_CACHE" \
      --garage-root "$GARAGE_ROOT" \
      --team-config "$TEAM_CONFIG" \
      --episode-root-override "$SHIFT_VIEW" \
      --cameras left,front,right \
      --tfpp-camera front \
      --command-mode target_angle \
      --image-size 640 360 \
      --lidar-size 128 \
      --batch-size "$CACHE_BATCH_SIZE" \
      --num-workers "$CACHE_WORKERS" \
      "${CACHE_ARGS[@]}"
  fi
fi

TRAIN_ARGS=()
if [[ "$DATA_PARALLEL" == "1" ]]; then
  TRAIN_ARGS+=(--data-parallel)
fi
if [[ -n "$TEACHER_CACHE" ]]; then
  TRAIN_ARGS+=(--teacher-cache "$TEACHER_CACHE")
fi

echo "=== train shifted visual/layout adapter"
PYTHONUNBUFFERED=1 "$PY" -m teach2drive_adapter.train_transfuserpp_cached_visual_adapter \
  --cache "$TRAIN_CACHE" \
  --index "$SHIFT_INDEX" \
  --episode-root-override "$SHIFT_VIEW" \
  --out-dir "$OUT" \
  --cameras left,front,right \
  --use-raw-layout \
  --image-size 320 180 \
  --lidar-size 128 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --moving-speed-threshold "$MOVING_SPEED_THRESHOLD" \
  --moving-sample-weight "$MOVING_SAMPLE_WEIGHT" \
  --stopped-sample-weight "$STOPPED_SAMPLE_WEIGHT" \
  --teacher-target-blend "$TEACHER_TARGET_BLEND" \
  --xy-loss-weight "$XY_LOSS_WEIGHT" \
  --yaw-loss-weight "$YAW_LOSS_WEIGHT" \
  --speed-loss-weight "$SPEED_LOSS_WEIGHT" \
  --speed-teacher-blend "$SPEED_TEACHER_BLEND" \
  --speed-distill-loss-weight "$SPEED_DISTILL_LOSS_WEIGHT" \
  --speed-floor-loss-weight "$SPEED_FLOOR_LOSS_WEIGHT" \
  --speed-floor-mps "$SPEED_FLOOR_MPS" \
  --speed-delta-loss-weight "$SPEED_DELTA_LOSS_WEIGHT" \
  --speed-curvature-loss-weight "$SPEED_CURVATURE_LOSS_WEIGHT" \
  --traj-delta-loss-weight "$TRAJ_DELTA_LOSS_WEIGHT" \
  --traj-curvature-loss-weight "$TRAJ_CURVATURE_LOSS_WEIGHT" \
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
  --stop-loss-weight "$STOP_LOSS_WEIGHT" \
  --stop-state-loss-weight "$STOP_STATE_LOSS_WEIGHT" \
  --stop-reason-loss-weight "$STOP_REASON_LOSS_WEIGHT" \
  --stop-positive-loss-scale "$STOP_POSITIVE_LOSS_SCALE" \
  --stop-negative-loss-scale "$STOP_NEGATIVE_LOSS_SCALE" \
  --stop-loss-after-epoch "$STOP_LOSS_AFTER_EPOCH" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$OUT/train.log"
