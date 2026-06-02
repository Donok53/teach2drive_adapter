#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Four cl1-based Town13 runs for improving closed-loop route completion.
# These keep the best observed direction so far:
#   - target-only expert labels, no TF++ output-prior loss;
#   - LoRA rank 8 cl1-style capacity;
#   - moderate stop/hazard weighting;
# and vary only go/progress/launch-release pressure plus checkpoint selection.

BASE_CONFIG="${BASE_CONFIG:-configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh}"
PY="${PY:-/home/jovyan_venv/.venv/torch2.1.2-py3.10-cuda11.8/bin/python}"
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

COMMON_CL1=(
  SEED=71
  LR=1.5e-5
  LORA_RANK=8
  LORA_ALPHA=16
  XY_LOSS_WEIGHT=0.55
  YAW_LOSS_WEIGHT=0.03
  SPEED_LOSS_WEIGHT=0.80
  CONTROL_LOSS_WEIGHT=0.85
  FEATURE_DRIFT_LOSS_WEIGHT=0.15
  STOPPED_SAMPLE_WEIGHT=1.7
  HAZARD_SAMPLE_WEIGHT=3.0
  LAUNCH_SAMPLE_WEIGHT=2.4
  RELEASE_SAMPLE_WEIGHT=2.2
  STOP_LOSS_WEIGHT=0.12
  STOP_SPEED_CEILING_LOSS_WEIGHT=0.60
  SPEED_FLOOR_LOSS_WEIGHT=0.07
  SPEED_FLOOR_MPS=0.9
  LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=0.12
  LAUNCH_SPEED_FLOOR_MPS=1.6
  RELEASE_SPEED_FLOOR_LOSS_WEIGHT=0.12
  RELEASE_SPEED_FLOOR_MPS=1.6
  STOP_STATE_AUX_LOSS_WEIGHT=0.15
  STOP_REASON_AUX_LOSS_WEIGHT=0.08
  TRAJ_SMOOTH_LOSS_WEIGHT=0.035
  SPEED_SMOOTH_LOSS_WEIGHT=0.020
)

launch action_cl1_go_floor_b24 0 \
  "${COMMON_CL1[@]}" \
  SEED=71 \
  SELECTION_METRIC=loss \
  SELECTION_MODE=min

launch action_cl1_go_proxy_select_b24 1 \
  "${COMMON_CL1[@]}" \
  SEED=72 \
  CONTROL_LOSS_WEIGHT=0.90 \
  GO_PROGRESS_RATIO=0.60 \
  SELECTION_METRIC=controller_closed_loop_proxy \
  SELECTION_MODE=min

launch action_cl1_turn_progress_b24 2 \
  "${COMMON_CL1[@]}" \
  SEED=73 \
  XY_LOSS_WEIGHT=0.62 \
  SPEED_LOSS_WEIGHT=0.85 \
  CONTROL_LOSS_WEIGHT=1.00 \
  SPEED_FLOOR_LOSS_WEIGHT=0.08 \
  LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=0.14 \
  RELEASE_SPEED_FLOOR_LOSS_WEIGHT=0.14 \
  GO_PROGRESS_RATIO=0.65 \
  SELECTION_METRIC=closed_loop_proxy \
  SELECTION_MODE=min

launch action_cl1_anti_stall_b24 3 \
  "${COMMON_CL1[@]}" \
  SEED=74 \
  STOPPED_SAMPLE_WEIGHT=1.5 \
  HAZARD_SAMPLE_WEIGHT=2.8 \
  LAUNCH_SAMPLE_WEIGHT=3.0 \
  RELEASE_SAMPLE_WEIGHT=2.8 \
  SPEED_FLOOR_LOSS_WEIGHT=0.10 \
  SPEED_FLOOR_MPS=1.1 \
  LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=0.20 \
  LAUNCH_SPEED_FLOOR_MPS=2.0 \
  RELEASE_SPEED_FLOOR_LOSS_WEIGHT=0.18 \
  RELEASE_SPEED_FLOOR_MPS=2.0 \
  CONTROL_LOSS_WEIGHT=1.10 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.12 \
  SELECTION_METRIC=loss \
  SELECTION_MODE=min

echo "sessions:"
tmux ls | grep -E 't2d_action_cl1_(go_floor|go_proxy_select|turn_progress|anti_stall)_b24' || true
