#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

# Follow-up training from the current best PDM-D progress/go checkpoint.
#
# All runs use only the target Tesla shifted-sensor 8-town dataset.  The
# checkpoint is used as the initialization anchor, not as a canonical-sensor
# teacher.  The goal is to preserve the early checkpoint behavior that worked
# in closed-loop while nudging a small number of failure modes.

BASE_CONFIG="${BASE_CONFIG:-configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh}"
PY="${PY:-/home/jovyan_venv/.venv/torch2.1.2-py3.10-cuda11.8/bin/python}"
BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT:-$HOME/teach2drive/workspace/teach2drive_bootstrap}"
PREP_ROOT="${PREP_ROOT:-$HOME/dataset/byeongjae/runs/tfpp_tesla_benchmix8_front_triplet_target_3h_prepared}"
RUN_ROOT="${RUN_ROOT:-$HOME/dataset/byeongjae/runs/benchmix8_pdm_d_epoch2_finetune4}"
LOG_DIR="${LOG_DIR:-$HOME/teach2drive/logs/benchmix8_pdm_d_epoch2_finetune4}"
PROFILE="${PROFILE:-front_triplet_shifted}"
CAMERAS="${CAMERAS:-left,front,right}"
TARGET_VIEW="${TARGET_VIEW:-$PREP_ROOT/profile_views/$PROFILE}"
TARGET_INDEX="${TARGET_INDEX:-$PREP_ROOT/indexes/${PROFILE}_index.npz}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$HOME/dataset/byeongjae/runs/benchmix8_pdm_d/pdm_d_progress_go_crop095_b24/best_model.pt}"

FT_GPU0="${FT_GPU0:-0}"
FT_GPU1="${FT_GPU1:-5}"
FT_GPU2="${FT_GPU2:-6}"
FT_GPU3="${FT_GPU3:-7}"

mkdir -p "$RUN_ROOT" "$LOG_DIR"

if [[ ! -f "$TARGET_INDEX" ]]; then
  echo "missing target index: $TARGET_INDEX" >&2
  exit 1
fi
if [[ ! -d "$TARGET_VIEW" ]]; then
  echo "missing target view: $TARGET_VIEW" >&2
  exit 1
fi
if [[ ! -f "$INIT_CHECKPOINT" ]]; then
  echo "missing init checkpoint: $INIT_CHECKPOINT" >&2
  exit 1
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

  echo "launch $name gpu=$gpu out=$out log=$log init=$INIT_CHECKPOINT"
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
    "INIT_CHECKPOINT=$INIT_CHECKPOINT"
    OVERWRITE=1
    SAVE_EPOCH_CHECKPOINTS=1
    EPOCH_CHECKPOINT_DIR=epoch_checkpoints
    BATCH_SIZE=24
    NUM_WORKERS=4
    OUTPUT_PRIOR_XY_LOSS_WEIGHT=0.0
    OUTPUT_PRIOR_SPEED_LOSS_WEIGHT=0.0
    SELECTION_METRIC=controller_closed_loop_proxy
    SELECTION_MODE=min
    CAMERA_CROP_SCALE=0.95
    CAMERA_CROP_SHIFT_X_PX=0.0
    CAMERA_CROP_SHIFT_Y_PX=0.0
    LIDAR_CANONICAL_SHIFT_X_M=0.0
    LIDAR_CANONICAL_SHIFT_Y_M=0.0
    STAGE_ADAPTER_MODALITIES=all
    FUSION_ADAPTER_ENABLED=1
    LORA_RANK=8
    LORA_ALPHA=16
    FEATURE_DRIFT_LOSS_WEIGHT=0.14
    STOP_LOSS_WEIGHT=0.14
    STOP_SPEED_CEILING_LOSS_WEIGHT=0.60
    SPEED_FLOOR_LOSS_WEIGHT=0.09
    LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=0.16
    RELEASE_SPEED_FLOOR_LOSS_WEIGHT=0.16
    STOP_STATE_AUX_LOSS_WEIGHT=0.15
    STOP_REASON_AUX_LOSS_WEIGHT=0.10
    TRAJ_SMOOTH_LOSS_WEIGHT=0.030
    SPEED_SMOOTH_LOSS_WEIGHT=0.020
    CONTROL_LOSS_WEIGHT=0.90
    "$@"
    bash "$BASE_CONFIG"
  )
  local command_string=""
  printf -v command_string "%q " "${cmd[@]}"
  tmux new-session -d -s "$session" -c "$PWD" -- bash -lc "$command_string > $(printf '%q' "$log") 2>&1"
}

# 1) Tiny continuation: keep the epoch-2 policy almost unchanged.
launch pdm_d_e2_tiny_lr3e6_b24 "$FT_GPU0" \
  SEED=121 \
  EPOCHS=8 \
  EARLY_STOP_PATIENCE=4 \
  LR=3.0e-6 \
  INIT_PARAM_ANCHOR_LOSS_WEIGHT=0.0 \
  STOPPED_SAMPLE_WEIGHT=1.6 \
  HAZARD_SAMPLE_WEIGHT=3.0 \
  LAUNCH_SAMPLE_WEIGHT=2.3 \
  RELEASE_SAMPLE_WEIGHT=2.1 \
  PDM_BEHAVIOR_LOSS_WEIGHT=0.12 \
  PDM_LATERAL_LOSS_WEIGHT=0.10 \
  PDM_PROGRESS_LOSS_WEIGHT=0.40 \
  PDM_HAZARD_PROGRESS_LOSS_WEIGHT=0.08 \
  PDM_CONTROLLER_LOSS_WEIGHT=0.25 \
  PDM_PLAN_STEER_LOSS_WEIGHT=0.06 \
  PDM_PLAN_THROTTLE_LOSS_WEIGHT=0.20 \
  PDM_PLAN_BRAKE_LOSS_WEIGHT=0.12

# 2) Teacher/anchor-style continuation: keep parameters close to epoch 2.
launch pdm_d_e2_anchor_lr5e6_b24 "$FT_GPU1" \
  SEED=122 \
  EPOCHS=8 \
  EARLY_STOP_PATIENCE=4 \
  LR=5.0e-6 \
  INIT_PARAM_ANCHOR_LOSS_WEIGHT=10.0 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.18 \
  STOPPED_SAMPLE_WEIGHT=1.6 \
  HAZARD_SAMPLE_WEIGHT=3.0 \
  LAUNCH_SAMPLE_WEIGHT=2.3 \
  RELEASE_SAMPLE_WEIGHT=2.1 \
  PDM_BEHAVIOR_LOSS_WEIGHT=0.12 \
  PDM_LATERAL_LOSS_WEIGHT=0.10 \
  PDM_PROGRESS_LOSS_WEIGHT=0.34 \
  PDM_HAZARD_PROGRESS_LOSS_WEIGHT=0.08 \
  PDM_CONTROLLER_LOSS_WEIGHT=0.24 \
  PDM_PLAN_STEER_LOSS_WEIGHT=0.06 \
  PDM_PLAN_THROTTLE_LOSS_WEIGHT=0.18 \
  PDM_PLAN_BRAKE_LOSS_WEIGHT=0.12

# 3) Failure-focused continuation: stronger hold/release/hazard pressure.
launch pdm_d_e2_failure_focus_b24 "$FT_GPU2" \
  SEED=123 \
  EPOCHS=8 \
  EARLY_STOP_PATIENCE=4 \
  LR=3.0e-6 \
  INIT_PARAM_ANCHOR_LOSS_WEIGHT=4.0 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.16 \
  STOPPED_SAMPLE_WEIGHT=2.2 \
  HAZARD_SAMPLE_WEIGHT=4.4 \
  LAUNCH_SAMPLE_WEIGHT=2.8 \
  RELEASE_SAMPLE_WEIGHT=2.6 \
  STOP_SPEED_CEILING_LOSS_WEIGHT=0.85 \
  PDM_BEHAVIOR_LOSS_WEIGHT=0.18 \
  PDM_LATERAL_LOSS_WEIGHT=0.12 \
  PDM_PROGRESS_LOSS_WEIGHT=0.30 \
  PDM_HAZARD_PROGRESS_LOSS_WEIGHT=0.18 \
  PDM_CONTROLLER_LOSS_WEIGHT=0.24 \
  PDM_PLAN_STEER_LOSS_WEIGHT=0.08 \
  PDM_PLAN_THROTTLE_LOSS_WEIGHT=0.18 \
  PDM_PLAN_BRAKE_LOSS_WEIGHT=0.20

# 4) Top-k / mini-selection candidate sweep: very low LR, more epochs saved.
launch pdm_d_e2_topk_sweep_lr15e7_b24 "$FT_GPU3" \
  SEED=124 \
  EPOCHS=10 \
  EARLY_STOP_PATIENCE=10 \
  LR=1.5e-6 \
  INIT_PARAM_ANCHOR_LOSS_WEIGHT=2.0 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.14 \
  STOPPED_SAMPLE_WEIGHT=1.8 \
  HAZARD_SAMPLE_WEIGHT=3.4 \
  LAUNCH_SAMPLE_WEIGHT=2.5 \
  RELEASE_SAMPLE_WEIGHT=2.3 \
  PDM_BEHAVIOR_LOSS_WEIGHT=0.12 \
  PDM_LATERAL_LOSS_WEIGHT=0.10 \
  PDM_PROGRESS_LOSS_WEIGHT=0.42 \
  PDM_HAZARD_PROGRESS_LOSS_WEIGHT=0.10 \
  PDM_CONTROLLER_LOSS_WEIGHT=0.24 \
  PDM_PLAN_STEER_LOSS_WEIGHT=0.06 \
  PDM_PLAN_THROTTLE_LOSS_WEIGHT=0.20 \
  PDM_PLAN_BRAKE_LOSS_WEIGHT=0.14

echo "sessions:"
tmux ls | grep -E 't2d_pdm_d_e2_' || true
