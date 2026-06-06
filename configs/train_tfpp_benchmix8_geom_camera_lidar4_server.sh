#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

# Four benchmark-aligned 8-town runs that test explicit input geometry
# canonicalization before learning behavior.  The dataset remains target-only:
# only front_triplet_shifted images/LiDAR are used.

BASE_CONFIG="${BASE_CONFIG:-configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh}"
PY="${PY:-/home/jovyan_venv/.venv/torch2.1.2-py3.10-cuda11.8/bin/python}"
BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT:-$HOME/teach2drive/workspace/teach2drive_bootstrap}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-$HOME/dataset/byeongjae/datasets/t2d_tesla_benchmix8_front_triplet_target_3h}"
PREP_ROOT="${PREP_ROOT:-$HOME/dataset/byeongjae/runs/tfpp_tesla_benchmix8_front_triplet_target_3h_prepared}"
RUN_ROOT="${RUN_ROOT:-$HOME/dataset/byeongjae/runs/benchmix8_geom_camera_lidar}"
LOG_DIR="${LOG_DIR:-$HOME/teach2drive/logs/benchmix8_geom_camera_lidar}"
PROFILE="${PROFILE:-front_triplet_shifted}"
CAMERAS="${CAMERAS:-left,front,right}"
TARGET_VIEW="${TARGET_VIEW:-$PREP_ROOT/profile_views/$PROFILE}"
TARGET_INDEX="${TARGET_INDEX:-$PREP_ROOT/indexes/${PROFILE}_index.npz}"
SKIP_PREP="${SKIP_PREP:-1}"
EXPORT_OVERWRITE="${EXPORT_OVERWRITE:-0}"
INDEX_OVERWRITE="${INDEX_OVERWRITE:-0}"

GEOM_GPU0="${GEOM_GPU0:-1}"
GEOM_GPU1="${GEOM_GPU1:-2}"
GEOM_GPU2="${GEOM_GPU2:-3}"
GEOM_GPU3="${GEOM_GPU3:-4}"

mkdir -p "$PREP_ROOT/profile_views" "$PREP_ROOT/indexes" "$RUN_ROOT" "$LOG_DIR"

if [[ "$SKIP_PREP" != "1" ]]; then
  echo "=== export $PROFILE profile view from $SOURCE_DATA_ROOT"
  export_args=()
  if [[ "$EXPORT_OVERWRITE" == "1" ]]; then
    export_args+=(--overwrite)
  fi
  "$PY" -m teach2drive_adapter.export_paired_profile_view \
    --input-root "$SOURCE_DATA_ROOT" \
    --output-root "$TARGET_VIEW" \
    --profile "$PROFILE" \
    --require-cameras "$CAMERAS" \
    --skip-invalid-motion \
    "${export_args[@]}"

  if [[ -f "$TARGET_INDEX" && "$INDEX_OVERWRITE" != "1" ]]; then
    echo "=== reuse index $TARGET_INDEX"
  else
    echo "=== build index $TARGET_INDEX cameras=$CAMERAS"
    (
      cd "$BOOTSTRAP_ROOT"
      PYTHONPATH="$BOOTSTRAP_ROOT:$REPO_ROOT:${PYTHONPATH:-}" "$PY" -m teach2drive.token_dataset \
        --input-root "$TARGET_VIEW" \
        --output "$TARGET_INDEX" \
        --cameras "$CAMERAS" \
        --augmentations 0 \
        --pseudo-label-name "__missing_pseudo_labels__.jsonl" \
        --seed 67
    )
  fi
fi

launch() {
  local name="$1"
  local gpu="$2"
  shift 2
  local session="t2d_${name}"
  local out="$RUN_ROOT/${name}"
  local log="$LOG_DIR/${name}.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session already exists: $session"
    return 0
  fi

  echo "launch $name gpu=$gpu out=$out log=$log"
  local cmd=()
  cmd+=(
    env
    "CUDA_VISIBLE_DEVICES=$gpu"
    "PY=$PY"
    PYTHONUNBUFFERED=1
    SNAPSHOT_COMPLETE_EPISODES=0
    REFRESH_SNAPSHOT=0
    SKIP_EXPORT=1
    INDEX_OVERWRITE=0
    "TARGET_VIEW=$TARGET_VIEW"
    "TARGET_INDEX=$TARGET_INDEX"
    "WORK_ROOT=$RUN_ROOT/${name}_work"
    "OUT=$out"
    OVERWRITE=1
    SAVE_EPOCH_CHECKPOINTS=1
    EPOCH_CHECKPOINT_DIR=epoch_checkpoints
    BATCH_SIZE=24
    NUM_WORKERS=4
    EPOCHS=18
    EARLY_STOP_PATIENCE=8
    EARLY_STOP_MIN_DELTA=0.0
    OUTPUT_PRIOR_XY_LOSS_WEIGHT=0.0
    OUTPUT_PRIOR_SPEED_LOSS_WEIGHT=0.0
    "$@"
    bash "$BASE_CONFIG"
  )
  local command_string=""
  printf -v command_string "%q " "${cmd[@]}"
  tmux new-session -d -s "$session" -c "$PWD" -- bash -lc "$command_string > $(printf '%q' "$log") 2>&1"
}

COMMON_ACTION=(
  STOPPED_SAMPLE_WEIGHT=1.7
  HAZARD_SAMPLE_WEIGHT=3.2
  LAUNCH_SAMPLE_WEIGHT=2.2
  RELEASE_SAMPLE_WEIGHT=2.0
  STOP_LOSS_WEIGHT=0.14
  STOP_SPEED_CEILING_LOSS_WEIGHT=0.65
  SPEED_FLOOR_LOSS_WEIGHT=0.05
  LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=0.08
  RELEASE_SPEED_FLOOR_LOSS_WEIGHT=0.08
  STOP_STATE_AUX_LOSS_WEIGHT=0.15
  STOP_REASON_AUX_LOSS_WEIGHT=0.08
  TRAJ_SMOOTH_LOSS_WEIGHT=0.035
  SPEED_SMOOTH_LOSS_WEIGHT=0.020
  CONTROL_LOSS_WEIGHT=0.90
  LR=1.2e-5
  LORA_RANK=8
  LORA_ALPHA=16
  FEATURE_DRIFT_LOSS_WEIGHT=0.12
  SELECTION_METRIC=controller_closed_loop_proxy
  SELECTION_MODE=min
)

launch geom_crop095_action_cl_b24 "$GEOM_GPU0" \
  "${COMMON_ACTION[@]}" \
  SEED=81 \
  CAMERA_CROP_SCALE=0.95 \
  CAMERA_CROP_SHIFT_Y_PX=0 \
  STAGE_ADAPTER_MODALITIES=all \
  FUSION_ADAPTER_ENABLED=1

launch geom_crop090_action_cl_b24 "$GEOM_GPU1" \
  "${COMMON_ACTION[@]}" \
  SEED=82 \
  CAMERA_CROP_SCALE=0.90 \
  CAMERA_CROP_SHIFT_Y_PX=0 \
  STAGE_ADAPTER_MODALITIES=all \
  FUSION_ADAPTER_ENABLED=1

launch geom_crop095_camera_early_b24 "$GEOM_GPU2" \
  "${COMMON_ACTION[@]}" \
  SEED=83 \
  CAMERA_CROP_SCALE=0.95 \
  CAMERA_CROP_SHIFT_Y_PX=0 \
  STAGE_ADAPTER_LAYERS=early:2 \
  STAGE_ADAPTER_MODALITIES=camera \
  FUSION_ADAPTER_ENABLED=0 \
  LORA_RANK=4 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.08

launch geom_crop095_lidar_shift_b24 "$GEOM_GPU3" \
  "${COMMON_ACTION[@]}" \
  SEED=84 \
  CAMERA_CROP_SCALE=0.95 \
  CAMERA_CROP_SHIFT_Y_PX=0 \
  LIDAR_CANONICAL_SHIFT_X_M=-0.20 \
  LIDAR_CANONICAL_SHIFT_Y_M=0.0 \
  STAGE_ADAPTER_MODALITIES=all \
  FUSION_ADAPTER_ENABLED=1

echo "sessions:"
tmux ls | grep -E 't2d_geom_crop' || true
