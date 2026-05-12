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

VIEW_ROOT=${VIEW_ROOT:-"$WORK_ROOT/profile_views"}
TFPP_VIEW=${TFPP_VIEW:-"$VIEW_ROOT/tfpp_ego"}
SHIFT_VIEW=${SHIFT_VIEW:-"$VIEW_ROOT/front_triplet_shifted"}
INDEX_DIR=${INDEX_DIR:-"$WORK_ROOT/indexes"}
TFPP_INDEX=${TFPP_INDEX:-"$INDEX_DIR/tfpp_ego_index.npz"}
SHIFT_INDEX=${SHIFT_INDEX:-"$INDEX_DIR/front_triplet_shifted_index.npz"}
CACHE=${CACHE:-"$WORK_ROOT/cache/tfpp_ego_prior_cache.npz"}
OUT=${OUT:-"$WORK_ROOT/train_shifted_visual_layout_adapter"}

mkdir -p "$WORK_ROOT" "$INDEX_DIR" "$(dirname "$CACHE")" "$OUT"

EXPORT_ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
  EXPORT_ARGS+=(--overwrite)
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

TRAIN_ARGS=()
if [[ "$DATA_PARALLEL" == "1" ]]; then
  TRAIN_ARGS+=(--data-parallel)
fi

echo "=== train shifted visual/layout adapter"
PYTHONUNBUFFERED=1 "$PY" -m teach2drive_adapter.train_transfuserpp_cached_visual_adapter \
  --cache "$CACHE" \
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
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$OUT/train.log"
