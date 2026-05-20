#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Feature-then-Fusion Adapter v1.
#
# This is NOT an output/policy adapter.
# It uses paired canonical/shifted data and trains:
#
#   1. stage feature adapters at each TransFuser++ fusion block input
#      layer_i image/lidar token maps:
#        shifted token map -> canonical-like token map
#   2. final fused-feature adapter:
#        shifted final fused feature -> canonical-like final fused feature
#
# The default DATA_ROOT is the older paired dataset because this experiment
# needs both tfpp_ego and front_triplet_shifted views for the same frames.

PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}
WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_feature_then_fusion_adapter_v1"}
BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-"$HOME/teach2drive/workspace/teach2drive_bootstrap"}
GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}
TFPP_CHECKPOINT=${TFPP_CHECKPOINT:-""}

VIEW_ROOT=${VIEW_ROOT:-"$WORK_ROOT/profile_views"}
TFPP_VIEW=${TFPP_VIEW:-"$VIEW_ROOT/tfpp_ego"}
SHIFT_VIEW=${SHIFT_VIEW:-"$VIEW_ROOT/front_triplet_shifted"}
INDEX_DIR=${INDEX_DIR:-"$WORK_ROOT/indexes"}
TFPP_INDEX=${TFPP_INDEX:-"$INDEX_DIR/tfpp_ego_index.npz"}
SHIFT_INDEX=${SHIFT_INDEX:-"$INDEX_DIR/front_triplet_shifted_index.npz"}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-"$WORK_ROOT/feature_fusion_cache"}
TFPP_FEATURE_CACHE=${TFPP_FEATURE_CACHE:-"$FEATURE_CACHE_DIR/tfpp_ego_feature_fusion"}
SHIFT_FEATURE_CACHE=${SHIFT_FEATURE_CACHE:-"$FEATURE_CACHE_DIR/front_triplet_shifted_feature_fusion"}
OUT=${OUT:-"$WORK_ROOT/train_feature_then_fusion_adapter_v1"}

AUGMENTATIONS=${AUGMENTATIONS:-0}
EXPORT_OVERWRITE=${EXPORT_OVERWRITE:-0}
INDEX_OVERWRITE=${INDEX_OVERWRITE:-0}
OVERWRITE=${OVERWRITE:-0}
SKIP_INVALID_MOTION=${SKIP_INVALID_MOTION:-1}
SEED=${SEED:-41}

FEATURE_CACHE_BATCH_SIZE=${FEATURE_CACHE_BATCH_SIZE:-32}
FEATURE_CACHE_WORKERS=${FEATURE_CACHE_WORKERS:-8}
FEATURE_CACHE_DTYPE=${FEATURE_CACHE_DTYPE:-float16}

EPOCHS=${EPOCHS:-50}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-10}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.0}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-2e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
VAL_RATIO=${VAL_RATIO:-0.15}
HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-0}
BLOCKS=${BLOCKS:-2}
DROPOUT=${DROPOUT:-0.0}
STAGE_LOSS_WEIGHT=${STAGE_LOSS_WEIGHT:-1.0}
FUSED_LOSS_WEIGHT=${FUSED_LOSS_WEIGHT:-1.0}
STAGE_FEATURE_LOSS_WEIGHT=${STAGE_FEATURE_LOSS_WEIGHT:-1.0}
STAGE_COSINE_LOSS_WEIGHT=${STAGE_COSINE_LOSS_WEIGHT:-0.05}
STAGE_RESIDUAL_LOSS_WEIGHT=${STAGE_RESIDUAL_LOSS_WEIGHT:-0.01}
FUSED_FEATURE_LOSS_WEIGHT=${FUSED_FEATURE_LOSS_WEIGHT:-1.0}
FUSED_COSINE_LOSS_WEIGHT=${FUSED_COSINE_LOSS_WEIGHT:-0.05}
FUSED_RESIDUAL_LOSS_WEIGHT=${FUSED_RESIDUAL_LOSS_WEIGHT:-0.01}
DATA_PARALLEL=${DATA_PARALLEL:-0}
TRAIN_ADAPTER=${TRAIN_ADAPTER:-1}

mkdir -p "$WORK_ROOT" "$INDEX_DIR" "$FEATURE_CACHE_DIR" "$OUT"

EXPORT_ARGS=()
if [[ "$EXPORT_OVERWRITE" == "1" ]]; then
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
  if [[ -f "$output" && "$INDEX_OVERWRITE" != "1" ]]; then
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

FEATURE_CACHE_ARGS=()
if [[ -n "$TFPP_CHECKPOINT" ]]; then
  FEATURE_CACHE_ARGS+=(--checkpoint "$TFPP_CHECKPOINT")
fi

if [[ -f "$TFPP_FEATURE_CACHE/fused.npy" && "$OVERWRITE" != "1" ]]; then
  echo "=== reuse canonical feature-fusion cache $TFPP_FEATURE_CACHE"
else
  echo "=== cache canonical tfpp_ego feature-fusion maps"
  rm -rf "$TFPP_FEATURE_CACHE"
  "$PY" -m teach2drive_adapter.cache_transfuserpp_feature_fusion_features \
    --index "$TFPP_INDEX" \
    --output-dir "$TFPP_FEATURE_CACHE" \
    --garage-root "$GARAGE_ROOT" \
    --team-config "$TEAM_CONFIG" \
    --episode-root-override "$TFPP_VIEW" \
    --cameras front \
    --tfpp-camera front \
    --command-mode target_angle \
    --image-size 640 360 \
    --lidar-size 128 \
    --batch-size "$FEATURE_CACHE_BATCH_SIZE" \
    --num-workers "$FEATURE_CACHE_WORKERS" \
    --dtype "$FEATURE_CACHE_DTYPE" \
    "${FEATURE_CACHE_ARGS[@]}"
fi

if [[ -f "$SHIFT_FEATURE_CACHE/fused.npy" && "$OVERWRITE" != "1" ]]; then
  echo "=== reuse shifted feature-fusion cache $SHIFT_FEATURE_CACHE"
else
  echo "=== cache shifted front_triplet feature-fusion maps"
  rm -rf "$SHIFT_FEATURE_CACHE"
  "$PY" -m teach2drive_adapter.cache_transfuserpp_feature_fusion_features \
    --index "$SHIFT_INDEX" \
    --output-dir "$SHIFT_FEATURE_CACHE" \
    --garage-root "$GARAGE_ROOT" \
    --team-config "$TEAM_CONFIG" \
    --episode-root-override "$SHIFT_VIEW" \
    --cameras left,front,right \
    --tfpp-camera front \
    --command-mode target_angle \
    --image-size 640 360 \
    --lidar-size 128 \
    --batch-size "$FEATURE_CACHE_BATCH_SIZE" \
    --num-workers "$FEATURE_CACHE_WORKERS" \
    --dtype "$FEATURE_CACHE_DTYPE" \
    "${FEATURE_CACHE_ARGS[@]}"
fi

if [[ "$TRAIN_ADAPTER" != "1" && "$TRAIN_ADAPTER" != "true" && "$TRAIN_ADAPTER" != "TRUE" ]]; then
  echo "=== skip feature-then-fusion adapter training; caches prepared"
  exit 0
fi

echo "=== train feature-then-fusion adapter v1"
TRAIN_ARGS=()
if [[ "$DATA_PARALLEL" == "1" || "$DATA_PARALLEL" == "true" || "$DATA_PARALLEL" == "TRUE" ]]; then
  TRAIN_ARGS+=(--data-parallel)
fi
PYTHONUNBUFFERED=1 "$PY" -m teach2drive_adapter.train_transfuserpp_feature_then_fusion_adapter \
  --source-cache "$SHIFT_FEATURE_CACHE" \
  --target-cache "$TFPP_FEATURE_CACHE" \
  --out-dir "$OUT" \
  --epochs "$EPOCHS" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --hidden-channels "$HIDDEN_CHANNELS" \
  --blocks "$BLOCKS" \
  --dropout "$DROPOUT" \
  --stage-loss-weight "$STAGE_LOSS_WEIGHT" \
  --fused-loss-weight "$FUSED_LOSS_WEIGHT" \
  --stage-feature-loss-weight "$STAGE_FEATURE_LOSS_WEIGHT" \
  --stage-cosine-loss-weight "$STAGE_COSINE_LOSS_WEIGHT" \
  --stage-residual-loss-weight "$STAGE_RESIDUAL_LOSS_WEIGHT" \
  --fused-feature-loss-weight "$FUSED_FEATURE_LOSS_WEIGHT" \
  --fused-cosine-loss-weight "$FUSED_COSINE_LOSS_WEIGHT" \
  --fused-residual-loss-weight "$FUSED_RESIDUAL_LOSS_WEIGHT" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$OUT/train.log"
