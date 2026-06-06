#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

# D-style training:
#   - train on the target-only 8-town Tesla shifted-sensor dataset
#   - include PDM/NAVSIM-style open-loop proxy terms directly in the loss
#   - save every epoch checkpoint for later mini closed-loop checkpoint selection
#
# No canonical/tfpp_ego sensor images are used.

BASE_CONFIG="${BASE_CONFIG:-configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh}"
PY="${PY:-/home/jovyan_venv/.venv/torch2.1.2-py3.10-cuda11.8/bin/python}"
BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT:-$HOME/teach2drive/workspace/teach2drive_bootstrap}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-$HOME/dataset/byeongjae/datasets/t2d_tesla_benchmix8_front_triplet_target_3h}"
PREP_ROOT="${PREP_ROOT:-$HOME/dataset/byeongjae/runs/tfpp_tesla_benchmix8_front_triplet_target_3h_prepared}"
RUN_ROOT="${RUN_ROOT:-$HOME/dataset/byeongjae/runs/benchmix8_pdm_d}"
LOG_DIR="${LOG_DIR:-$HOME/teach2drive/logs/benchmix8_pdm_d}"
PROFILE="${PROFILE:-front_triplet_shifted}"
CAMERAS="${CAMERAS:-left,front,right}"
TARGET_VIEW="${TARGET_VIEW:-$PREP_ROOT/profile_views/$PROFILE}"
TARGET_INDEX="${TARGET_INDEX:-$PREP_ROOT/indexes/${PROFILE}_index.npz}"
SKIP_PREP="${SKIP_PREP:-1}"
EXPORT_OVERWRITE="${EXPORT_OVERWRITE:-0}"
INDEX_OVERWRITE="${INDEX_OVERWRITE:-0}"

PDM_GPU0="${PDM_GPU0:-1}"
PDM_GPU1="${PDM_GPU1:-2}"
PDM_GPU2="${PDM_GPU2:-3}"
PDM_GPU3="${PDM_GPU3:-4}"

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
        --seed 91
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
    EPOCHS=20
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

COMMON_D=(
  SELECTION_METRIC=controller_closed_loop_proxy
  SELECTION_MODE=min
  CAMERA_CROP_SCALE=0.95
  CAMERA_CROP_SHIFT_X_PX=0.0
  CAMERA_CROP_SHIFT_Y_PX=0.0
  LIDAR_CANONICAL_SHIFT_X_M=0.0
  LIDAR_CANONICAL_SHIFT_Y_M=0.0
  STAGE_ADAPTER_MODALITIES=all
  FUSION_ADAPTER_ENABLED=1
  LR=1.0e-5
  LORA_RANK=8
  LORA_ALPHA=16
  FEATURE_DRIFT_LOSS_WEIGHT=0.12
  STOPPED_SAMPLE_WEIGHT=1.6
  HAZARD_SAMPLE_WEIGHT=3.0
  LAUNCH_SAMPLE_WEIGHT=2.3
  RELEASE_SAMPLE_WEIGHT=2.1
  STOP_LOSS_WEIGHT=0.14
  STOP_SPEED_CEILING_LOSS_WEIGHT=0.60
  SPEED_FLOOR_LOSS_WEIGHT=0.06
  LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=0.10
  RELEASE_SPEED_FLOOR_LOSS_WEIGHT=0.10
  STOP_STATE_AUX_LOSS_WEIGHT=0.15
  STOP_REASON_AUX_LOSS_WEIGHT=0.10
  TRAJ_SMOOTH_LOSS_WEIGHT=0.030
  SPEED_SMOOTH_LOSS_WEIGHT=0.020
  CONTROL_LOSS_WEIGHT=0.90
)

launch pdm_d_balanced_crop095_b24 "$PDM_GPU0" \
  "${COMMON_D[@]}" \
  SEED=91 \
  PDM_BEHAVIOR_LOSS_WEIGHT=0.18 \
  PDM_LATERAL_LOSS_WEIGHT=0.16 \
  PDM_PROGRESS_LOSS_WEIGHT=0.22 \
  PDM_HAZARD_PROGRESS_LOSS_WEIGHT=0.12 \
  PDM_CONTROLLER_LOSS_WEIGHT=0.25 \
  PDM_PLAN_STEER_LOSS_WEIGHT=0.08 \
  PDM_PLAN_THROTTLE_LOSS_WEIGHT=0.12 \
  PDM_PLAN_BRAKE_LOSS_WEIGHT=0.18

launch pdm_d_progress_go_crop095_b24 "$PDM_GPU1" \
  "${COMMON_D[@]}" \
  SEED=92 \
  SPEED_FLOOR_LOSS_WEIGHT=0.09 \
  LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=0.16 \
  RELEASE_SPEED_FLOOR_LOSS_WEIGHT=0.16 \
  PDM_BEHAVIOR_LOSS_WEIGHT=0.12 \
  PDM_LATERAL_LOSS_WEIGHT=0.10 \
  PDM_PROGRESS_LOSS_WEIGHT=0.40 \
  PDM_HAZARD_PROGRESS_LOSS_WEIGHT=0.08 \
  PDM_CONTROLLER_LOSS_WEIGHT=0.25 \
  PDM_PLAN_STEER_LOSS_WEIGHT=0.06 \
  PDM_PLAN_THROTTLE_LOSS_WEIGHT=0.20 \
  PDM_PLAN_BRAKE_LOSS_WEIGHT=0.12

launch pdm_d_safety_crop095_b24 "$PDM_GPU2" \
  "${COMMON_D[@]}" \
  SEED=93 \
  STOP_SPEED_CEILING_LOSS_WEIGHT=0.80 \
  HAZARD_SAMPLE_WEIGHT=3.8 \
  PDM_BEHAVIOR_LOSS_WEIGHT=0.22 \
  PDM_LATERAL_LOSS_WEIGHT=0.22 \
  PDM_PROGRESS_LOSS_WEIGHT=0.16 \
  PDM_HAZARD_PROGRESS_LOSS_WEIGHT=0.22 \
  PDM_CONTROLLER_LOSS_WEIGHT=0.20 \
  PDM_PLAN_STEER_LOSS_WEIGHT=0.10 \
  PDM_PLAN_THROTTLE_LOSS_WEIGHT=0.08 \
  PDM_PLAN_BRAKE_LOSS_WEIGHT=0.22

launch pdm_d_controller_crop095_b24 "$PDM_GPU3" \
  "${COMMON_D[@]}" \
  SEED=94 \
  CONTROL_LOSS_WEIGHT=1.20 \
  PDM_BEHAVIOR_LOSS_WEIGHT=0.12 \
  PDM_LATERAL_LOSS_WEIGHT=0.12 \
  PDM_PROGRESS_LOSS_WEIGHT=0.20 \
  PDM_HAZARD_PROGRESS_LOSS_WEIGHT=0.10 \
  PDM_CONTROLLER_LOSS_WEIGHT=0.45 \
  PDM_PLAN_STEER_LOSS_WEIGHT=0.12 \
  PDM_PLAN_THROTTLE_LOSS_WEIGHT=0.20 \
  PDM_PLAN_BRAKE_LOSS_WEIGHT=0.28

echo "sessions:"
tmux ls | grep -E 't2d_pdm_d_' || true
