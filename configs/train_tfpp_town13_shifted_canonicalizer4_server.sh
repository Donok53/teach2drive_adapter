#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Four shifted-sensor-only canonicalizer runs.
#
# Important: these runs use only front_triplet_shifted target-rig data.
# They do not export, cache, or supervise from tfpp_ego/canonical sensor views.
# "distill" here means conservative internal feature regularization
# (feature drift), with expert labels remaining the driving target.

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

COMMON_EXPERT=(
  EXTRINSIC_AWARE=1
  SOURCE_PROFILE=front_triplet_shifted
  STAGE_FEATURE_ADAPTER_BLEND=1.0
  FUSION_ADAPTER_BLEND=1.0
  STOPPED_SAMPLE_WEIGHT=1.7
  HAZARD_SAMPLE_WEIGHT=3.0
  LAUNCH_SAMPLE_WEIGHT=2.2
  RELEASE_SAMPLE_WEIGHT=2.0
  STOP_LOSS_WEIGHT=0.12
  STOP_SPEED_CEILING_LOSS_WEIGHT=0.55
  SPEED_FLOOR_LOSS_WEIGHT=0.06
  LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=0.10
  RELEASE_SPEED_FLOOR_LOSS_WEIGHT=0.10
  STOP_STATE_AUX_LOSS_WEIGHT=0.12
  STOP_REASON_AUX_LOSS_WEIGHT=0.06
  TRAJ_SMOOTH_LOSS_WEIGHT=0.030
  SPEED_SMOOTH_LOSS_WEIGHT=0.018
)

# A. Camera-stage canonicalizer only + weak expert BC.
launch canon_a_camera_feature_distill_b24 0 \
  "${COMMON_EXPERT[@]}" \
  SEED=81 \
  LR=1.5e-5 \
  STAGE_ADAPTER_LAYERS=all \
  STAGE_ADAPTER_MODALITIES=camera \
  FUSION_ADAPTER_ENABLED=0 \
  LORA_RANK=0 \
  XY_LOSS_WEIGHT=0.35 \
  SPEED_LOSS_WEIGHT=0.45 \
  CONTROL_LOSS_WEIGHT=0.25 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.30 \
  SELECTION_METRIC=loss \
  SELECTION_MODE=min

# B. Camera+LiDAR stage plus fused canonicalizer + weak expert BC.
launch canon_b_fused_feature_distill_b24 1 \
  "${COMMON_EXPERT[@]}" \
  SEED=82 \
  LR=1.2e-5 \
  STAGE_ADAPTER_LAYERS=all \
  STAGE_ADAPTER_MODALITIES=all \
  FUSION_ADAPTER_ENABLED=1 \
  LORA_RANK=0 \
  XY_LOSS_WEIGHT=0.40 \
  SPEED_LOSS_WEIGHT=0.55 \
  CONTROL_LOSS_WEIGHT=0.35 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.22 \
  SELECTION_METRIC=loss \
  SELECTION_MODE=min

# C. Early canonicalizer only; decoder/head fully frozen.
launch canon_c_early_frozen_head_b24 2 \
  "${COMMON_EXPERT[@]}" \
  SEED=83 \
  LR=1.5e-5 \
  STAGE_ADAPTER_LAYERS=early:2 \
  STAGE_ADAPTER_MODALITIES=all \
  FUSION_ADAPTER_ENABLED=0 \
  LORA_RANK=0 \
  XY_LOSS_WEIGHT=0.55 \
  SPEED_LOSS_WEIGHT=0.80 \
  CONTROL_LOSS_WEIGHT=0.80 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.18 \
  SELECTION_METRIC=controller_closed_loop_proxy \
  SELECTION_MODE=min

# D. Early canonicalizer plus tiny LoRA in TF++ planning heads.
launch canon_d_early_lora4_head_b24 3 \
  "${COMMON_EXPERT[@]}" \
  SEED=84 \
  LR=1.0e-5 \
  STAGE_ADAPTER_LAYERS=early:2 \
  STAGE_ADAPTER_MODALITIES=all \
  FUSION_ADAPTER_ENABLED=0 \
  LORA_RANK=4 \
  LORA_ALPHA=8 \
  XY_LOSS_WEIGHT=0.55 \
  SPEED_LOSS_WEIGHT=0.80 \
  CONTROL_LOSS_WEIGHT=0.85 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.16 \
  SELECTION_METRIC=controller_closed_loop_proxy \
  SELECTION_MODE=min

echo "sessions:"
tmux ls | grep -E 't2d_canon_[abcd]_' || true
