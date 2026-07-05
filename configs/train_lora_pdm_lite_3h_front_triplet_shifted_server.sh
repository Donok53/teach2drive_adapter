#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ==============================================================================
# Single-run LoRA adaptation on the pdm_lite 3h EVEN subset
#   ego vehicle : vehicle.tesla.model3   (changed vehicle)
#   sensor rig  : front_triplet_shifted  (changed sensor position, 3 cameras)
#   base policy : frozen TransFuser++, adapted with LoRA on
#                 join / checkpoint_decoder / target_speed_network
#
# Goal: prove a pretrained E2E policy adapts to the changed vehicle+sensor
#       geometry from only ~3h of driving via LoRA, evaluated closed-loop.
#
# DATA DEPENDENCY (IMPORTANT):
#   SOURCE_DATA_ROOT must be the t2d-token version of the 3h subset
#   (frames.jsonl + profile_tokens + rigs/<profile>/sensor_layout.json).
#   The raw carla_garage copy lives at:
#     ~/dataset/byeongjae/datasets/pdm_lite_front_triplet_shifted_3h_subset
#   It must first be converted to t2d episodes at SOURCE_DATA_ROOT below.
#   (converter is the next prep step; this config is ready to run once it exists.)
# ==============================================================================

PY=${PY:-python}
REPO_ROOT="$PWD"
BASE_CONFIG=${BASE_CONFIG:-configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh}
BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-$HOME/teach2drive/workspace/teach2drive_bootstrap}

# --- data (t2d-converted 3h subset) ---
SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-$HOME/dataset/byeongjae/datasets/t2d_pdm_lite_front_triplet_shifted_3h}
PREP_ROOT=${PREP_ROOT:-$HOME/dataset/byeongjae/runs/lora_pdm_lite_3h_prepared}
RUN_ROOT=${RUN_ROOT:-$HOME/dataset/byeongjae/runs/lora_pdm_lite_3h_front_triplet_shifted}
LOG_DIR=${LOG_DIR:-$HOME/teach2drive/logs}
PROFILE=${PROFILE:-front_triplet_shifted}
CAMERAS=${CAMERAS:-left,front,right}
TARGET_VIEW=${TARGET_VIEW:-$PREP_ROOT/profile_views/$PROFILE}
TARGET_INDEX=${TARGET_INDEX:-$PREP_ROOT/indexes/${PROFILE}_index.npz}
GPU=${GPU:-0}

mkdir -p "$PREP_ROOT/profile_views" "$PREP_ROOT/indexes" "$RUN_ROOT" "$LOG_DIR"

# --- 1) export profile view + build token index (once) --------------------------
if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  echo "=== export $PROFILE profile view from $SOURCE_DATA_ROOT"
  export_args=()
  [[ "${EXPORT_OVERWRITE:-0}" == "1" ]] && export_args+=(--overwrite)
  "$PY" -m teach2drive_adapter.export_paired_profile_view \
    --input-root "$SOURCE_DATA_ROOT" \
    --output-root "$TARGET_VIEW" \
    --profile "$PROFILE" \
    --require-cameras "$CAMERAS" \
    --skip-invalid-motion \
    "${export_args[@]}"

  if [[ -f "$TARGET_INDEX" && "${INDEX_OVERWRITE:-0}" != "1" ]]; then
    echo "=== reuse index $TARGET_INDEX"
  else
    echo "=== build token index $TARGET_INDEX cameras=$CAMERAS"
    ( cd "$BOOTSTRAP_ROOT"
      PYTHONPATH="$BOOTSTRAP_ROOT:$REPO_ROOT:${PYTHONPATH:-}" "$PY" -m teach2drive.token_dataset \
        --input-root "$TARGET_VIEW" \
        --output "$TARGET_INDEX" \
        --cameras "$CAMERAS" \
        --augmentations 0 \
        --pseudo-label-name "__missing_pseudo_labels__.jsonl" \
        --seed 91 )
  fi
fi

# --- 2) LoRA training (frozen TF++ backbone; LoRA on fusion/planner) -------------
echo "=== LoRA training -> $RUN_ROOT/train_${PROFILE}_lora"
env \
  CUDA_VISIBLE_DEVICES="$GPU" PY="$PY" PYTHONUNBUFFERED=1 \
  SNAPSHOT_COMPLETE_EPISODES=0 REFRESH_SNAPSHOT=0 SKIP_EXPORT=1 INDEX_OVERWRITE=0 \
  TARGET_VIEW="$TARGET_VIEW" TARGET_INDEX="$TARGET_INDEX" \
  WORK_ROOT="$RUN_ROOT/work" OUT="$RUN_ROOT/train_${PROFILE}_lora" OVERWRITE=1 \
  SAVE_EPOCH_CHECKPOINTS=1 EPOCH_CHECKPOINT_DIR=epoch_checkpoints \
  BATCH_SIZE=${BATCH_SIZE:-24} NUM_WORKERS=${NUM_WORKERS:-4} \
  EPOCHS=${EPOCHS:-20} EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-8} EARLY_STOP_MIN_DELTA=0.0 \
  SELECTION_METRIC=${SELECTION_METRIC:-loss} SELECTION_MODE=min \
  LR=${LR:-1.0e-5} WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4} \
  LORA_RANK=${LORA_RANK:-8} LORA_ALPHA=${LORA_ALPHA:-16} LORA_DROPOUT=${LORA_DROPOUT:-0.02} \
  LORA_INCLUDE='^join\.,^checkpoint_decoder\.(encoder|decoder)\.,^target_speed_network\.' \
  EXTRINSIC_AWARE=1 SOURCE_PROFILE="$PROFILE" \
  FUSION_ADAPTER_ENABLED=1 STAGE_ADAPTER_MODALITIES=all \
  FEATURE_DRIFT_LOSS_WEIGHT=0.12 \
  STOP_LOSS_WEIGHT=0.14 STOP_SPEED_CEILING_LOSS_WEIGHT=0.60 \
  STOPPED_SAMPLE_WEIGHT=1.6 HAZARD_SAMPLE_WEIGHT=3.0 \
  LAUNCH_SAMPLE_WEIGHT=2.3 RELEASE_SAMPLE_WEIGHT=2.1 \
  SPEED_FLOOR_LOSS_WEIGHT=0.06 LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=0.10 RELEASE_SPEED_FLOOR_LOSS_WEIGHT=0.10 \
  STOP_STATE_AUX_LOSS_WEIGHT=0.15 STOP_REASON_AUX_LOSS_WEIGHT=0.10 \
  TRAJ_SMOOTH_LOSS_WEIGHT=0.030 SPEED_SMOOTH_LOSS_WEIGHT=0.020 CONTROL_LOSS_WEIGHT=0.90 \
  SEED=${SEED:-91} \
  bash "$BASE_CONFIG"

echo "=== done. checkpoint dir: $RUN_ROOT/train_${PROFILE}_lora"
