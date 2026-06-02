#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Four parallel target-only adapter runs for the "open-loop proxy is not enough"
# stage.  These runs all:
#   - train against expert labels, not TF++ prior outputs;
#   - enable expert control supervision;
#   - upweight stop/go, hazard, launch/release cases;
#   - save every epoch checkpoint so a later mini closed-loop validation can
#     select the deployable checkpoint instead of trusting one open-loop metric.

BASE_CONFIG="${BASE_CONFIG:-configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh}"
PY="${PY:-/home/jovyan_venv/.venv/torch2.3.0-py3.10-cuda11.8/bin/python}"
LOG_DIR="${LOG_DIR:-$HOME/teach2drive/logs}"
RUN_ROOT="${RUN_ROOT:-$HOME/dataset/byeongjae/runs}"
VIEW_ROOT="${VIEW_ROOT:-$RUN_ROOT/tfpp_tesla_town13_expert_only_lora16_head_blend1/profile_views}"
INDEX_DIR="${INDEX_DIR:-$RUN_ROOT/tfpp_tesla_town13_expert_only_lora16_head_blend1/indexes}"
TARGET_VIEW="${TARGET_VIEW:-$VIEW_ROOT/front_triplet_shifted}"
TARGET_INDEX="${TARGET_INDEX:-$INDEX_DIR/front_triplet_shifted_index.npz}"

mkdir -p "$LOG_DIR"

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

COMMON_RISK=(
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
)

launch action_cl1_ctrl_lora8_b24 0 \
  "${COMMON_RISK[@]}" \
  SEED=61 \
  LR=1.5e-5 \
  LORA_RANK=8 \
  LORA_ALPHA=16 \
  CONTROL_LOSS_WEIGHT=0.80 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.15 \
  SELECTION_METRIC=loss \
  SELECTION_MODE=min

launch action_cl2_ctrl_lora16_b24 1 \
  "${COMMON_RISK[@]}" \
  SEED=62 \
  LR=1.0e-5 \
  LORA_RANK=16 \
  LORA_ALPHA=32 \
  CONTROL_LOSS_WEIGHT=1.00 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.12 \
  SELECTION_METRIC=loss \
  SELECTION_MODE=min

launch action_cl3_hazard_ctrl_b24 2 \
  "${COMMON_RISK[@]}" \
  SEED=63 \
  LR=1.5e-5 \
  LORA_RANK=8 \
  LORA_ALPHA=16 \
  CONTROL_LOSS_WEIGHT=1.20 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.18 \
  HAZARD_SAMPLE_WEIGHT=4.2 \
  STOPPED_SAMPLE_WEIGHT=2.0 \
  STOP_SPEED_CEILING_LOSS_WEIGHT=0.85 \
  SELECTION_METRIC=controller_closed_loop_proxy \
  SELECTION_MODE=min

launch action_cl4_conservative_ctrl_b24 3 \
  "${COMMON_RISK[@]}" \
  SEED=64 \
  LR=8e-6 \
  LORA_RANK=4 \
  LORA_ALPHA=8 \
  CONTROL_LOSS_WEIGHT=1.00 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.25 \
  XY_LOSS_WEIGHT=0.50 \
  SPEED_LOSS_WEIGHT=0.70 \
  SELECTION_METRIC=controller_closed_loop_proxy \
  SELECTION_MODE=min

echo "sessions:"
tmux ls | grep -E 't2d_action_cl[1-4]|t2d_action_cl[23]_|t2d_action_cl4_' || true
