#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Four 8-town shifted-sensor-only front-end adaptation runs.
#
# These runs use only the front_triplet_shifted target rig:
#   /datasets/t2d_tesla_benchmix8_front_triplet_target_3h
#
# No tfpp_ego/canonical sensor images are exported, cached, or supervised.
# The "canonical" wording here means: make shifted front-end features easier for
# the frozen TransFuser++ policy to consume, using expert BC/control labels.

BASE_CONFIG="${BASE_CONFIG:-configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh}"
PY="${PY:-/home/jovyan_venv/.venv/torch2.1.2-py3.10-cuda11.8/bin/python}"
LOG_DIR="${LOG_DIR:-$HOME/teach2drive/logs}"
RUN_ROOT="${RUN_ROOT:-$HOME/dataset/byeongjae/runs}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$RUN_ROOT/benchmix8_frontend_canonical_abcd}"
PREP_ROOT="${PREP_ROOT:-$RUN_ROOT/tfpp_tesla_benchmix8_front_triplet_target_3h_prepared}"
TARGET_VIEW="${TARGET_VIEW:-$PREP_ROOT/profile_views/front_triplet_shifted}"
TARGET_INDEX="${TARGET_INDEX:-$PREP_ROOT/indexes/front_triplet_shifted_index.npz}"

mkdir -p "$LOG_DIR" "$EXPERIMENT_ROOT"

launch() {
  local name="$1"
  local gpu="$2"
  shift 2
  local session="t2d_${name}"
  local out="$EXPERIMENT_ROOT/${name}"
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
    "WORK_ROOT=$EXPERIMENT_ROOT/${name}_work"
    "OUT=$out"
    OVERWRITE=1
    BATCH_SIZE=24
    NUM_WORKERS=4
    EPOCHS=18
    EARLY_STOP_PATIENCE=8
    EARLY_STOP_MIN_DELTA=0.0
    OUTPUT_PRIOR_XY_LOSS_WEIGHT=0.0
    OUTPUT_PRIOR_SPEED_LOSS_WEIGHT=0.0
    EXTRINSIC_AWARE=1
    SOURCE_PROFILE=front_triplet_shifted
    STAGE_FEATURE_ADAPTER_BLEND=1.0
    FUSION_ADAPTER_BLEND=1.0
    STOPPED_SAMPLE_WEIGHT=1.55
    HAZARD_SAMPLE_WEIGHT=2.7
    LAUNCH_SAMPLE_WEIGHT=2.0
    RELEASE_SAMPLE_WEIGHT=1.8
    STOP_LOSS_WEIGHT=0.10
    STOP_SPEED_CEILING_LOSS_WEIGHT=0.50
    SPEED_FLOOR_LOSS_WEIGHT=0.055
    LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=0.09
    RELEASE_SPEED_FLOOR_LOSS_WEIGHT=0.09
    STOP_STATE_AUX_LOSS_WEIGHT=0.10
    STOP_REASON_AUX_LOSS_WEIGHT=0.05
    TRAJ_SMOOTH_LOSS_WEIGHT=0.025
    SPEED_SMOOTH_LOSS_WEIGHT=0.012
    "$@"
    bash "$BASE_CONFIG"
  )
  local command_string=""
  printf -v command_string "%q " "${cmd[@]}"
  tmux new-session -d -s "$session" -c "$PWD" -- bash -lc "$command_string > $(printf '%q' "$log") 2>&1"
}

# A. Explicit LiDAR BEV geometry correction + camera early adapter + expert BC.
launch benchmix8_geom_lidar_cam_early_b24 0 \
  SEED=101 \
  LR=1.5e-5 \
  LIDAR_CANONICAL_SHIFT_X_M=-0.20 \
  LIDAR_CANONICAL_SHIFT_Y_M=0.0 \
  STAGE_ADAPTER_LAYERS=early:2 \
  STAGE_ADAPTER_MODALITIES=camera \
  FUSION_ADAPTER_ENABLED=0 \
  LORA_RANK=0 \
  XY_LOSS_WEIGHT=0.45 \
  SPEED_LOSS_WEIGHT=0.65 \
  CONTROL_LOSS_WEIGHT=0.30 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.16 \
  SELECTION_METRIC=loss \
  SELECTION_MODE=min

# B. Camera+LiDAR early encoder fine-tune; TF++ head/decoder stays frozen.
launch benchmix8_early_unfreeze_cam_lidar_b24 1 \
  SEED=102 \
  LR=1.2e-5 \
  UNFREEZE_INCLUDE='^backbone\\.(image_encoder|lidar_encoder)\\.(stem|s1|s2)' \
  UNFREEZE_LR=2.0e-6 \
  UNFREEZE_WEIGHT_DECAY=1e-6 \
  STAGE_ADAPTER_LAYERS=early:2 \
  STAGE_ADAPTER_MODALITIES=all \
  FUSION_ADAPTER_ENABLED=0 \
  LORA_RANK=0 \
  XY_LOSS_WEIGHT=0.55 \
  SPEED_LOSS_WEIGHT=0.75 \
  CONTROL_LOSS_WEIGHT=0.45 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.12 \
  SELECTION_METRIC=controller_closed_loop_proxy \
  SELECTION_MODE=min

# C. Front-end adapter, expert BC loss, select by controller/progress proxy.
launch benchmix8_frontend_ctrl_progress_b24 2 \
  SEED=103 \
  LR=1.2e-5 \
  STAGE_ADAPTER_LAYERS=early:3 \
  STAGE_ADAPTER_MODALITIES=all \
  FUSION_ADAPTER_ENABLED=0 \
  LORA_RANK=0 \
  XY_LOSS_WEIGHT=0.55 \
  SPEED_LOSS_WEIGHT=0.80 \
  CONTROL_LOSS_WEIGHT=0.90 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.10 \
  GO_PROGRESS_RATIO=0.65 \
  CONTROLLER_THROTTLE_CLOSE_THRESHOLD=0.16 \
  SELECTION_METRIC=controller_closed_loop_proxy \
  SELECTION_MODE=min

# D. Front-end adapter with epoch checkpoints for later mini closed-loop choice.
launch benchmix8_frontend_epoch_ckpt_b24 3 \
  SEED=104 \
  LR=1.0e-5 \
  SAVE_EPOCH_CHECKPOINTS=1 \
  EPOCH_CHECKPOINT_DIR=epoch_checkpoints \
  EARLY_STOP_PATIENCE=18 \
  STAGE_ADAPTER_LAYERS=early:3 \
  STAGE_ADAPTER_MODALITIES=all \
  FUSION_ADAPTER_ENABLED=0 \
  LORA_RANK=4 \
  LORA_ALPHA=8 \
  XY_LOSS_WEIGHT=0.55 \
  SPEED_LOSS_WEIGHT=0.80 \
  CONTROL_LOSS_WEIGHT=0.85 \
  FEATURE_DRIFT_LOSS_WEIGHT=0.08 \
  SELECTION_METRIC=controller_closed_loop_proxy \
  SELECTION_MODE=min

echo "sessions:"
tmux ls | grep -E 't2d_benchmix8_(geom_lidar|early_unfreeze|frontend_ctrl|frontend_epoch)' || true
