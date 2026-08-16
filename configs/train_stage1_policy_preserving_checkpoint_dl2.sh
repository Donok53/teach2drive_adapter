#!/usr/bin/env bash
set -euo pipefail

# Stage 1A: preserve pretrained TF++ and learn only a small lateral checkpoint
# residual on target-domain turns.  PDM future poses are a correction signal,
# not a replacement policy.  Target-speed logits and the original PID stay fixed.

export RUN_NAME=${RUN_NAME:-train_stage1_policy_preserving_checkpoint_gpu0}
export TEACHER_VIEW_ROOT=${TEACHER_VIEW_ROOT:-/data/dataset/byeongjae/datasets/pdm_lite_tesla_paired_3h/data}
export TEACHER_VIEW_AS_INPUT=1

export STAGE_ADAPTER_LAYERS=all
export STAGE_ADAPTER_MODALITIES=none
export DISABLE_FUSION_ADAPTER=1
export FREEZE_ADAPTER_INCLUDE='.*'
export FEATURE_DRIFT_WEIGHT=0.0
export EXTRINSIC_AWARE=0
export SOURCE_PROFILE=tfpp_ego
export LORA_RANK=0
export LORA_ALPHA=1
export LORA_INCLUDE=''
export UNFREEZE_INCLUDE=''

export OUTPUT_RESIDUAL=1
export OUTPUT_RESIDUAL_CHECKPOINT_LATERAL_ONLY=1
export OUTPUT_RESIDUAL_HIDDEN_DIM=128
export OUTPUT_RESIDUAL_CHECKPOINT_SCALE=${OUTPUT_RESIDUAL_CHECKPOINT_SCALE:-0.15}
export OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE=0.0
export OUTPUT_RESIDUAL_GATE_BIAS=${OUTPUT_RESIDUAL_GATE_BIAS:--3.0}
export OUTPUT_RESIDUAL_DROPOUT=0.0

export ROUTE_TARGET_ONLY=1
export ROUTE_TARGET_SOURCE=future_ego_path
export ROUTE_TARGET_LEN=10
export FUTURE_EGO_MAX_HORIZON_S=6.0
export ROUTE_LATERAL_LOSS_WEIGHT=1.0
export ROUTE_HEADING_LOSS_WEIGHT=0.10
export ROUTE_AIM_ANGLE_LOSS_WEIGHT=2.0
export ROUTE_STRAIGHT_IDENTITY_LOSS_WEIGHT=1.0
export ROUTE_TURN_GATE_LOSS_WEIGHT=0.10
export ROUTE_TURN_LATERAL_THRESHOLD_M=0.50
export ROUTE_TURN_ANGLE_THRESHOLD_RAD=0.08

# This prior is intentionally active inside ROUTE_TARGET_ONLY.  It anchors the
# adapted trajectory to the original TF++ trajectory on both straight and turns.
export OUTPUT_PRIOR_XY_LOSS_WEIGHT=${OUTPUT_PRIOR_XY_LOSS_WEIGHT:-5.0}
export OUTPUT_PRIOR_SPEED_LOSS_WEIGHT=0.0
export CONTROL_LOSS_WEIGHT=0.0
export STOP_STATE_AUX_LOSS_WEIGHT=0.0
export STOP_REASON_AUX_LOSS_WEIGHT=0.0
export PDM_BEHAVIOR_LOSS_WEIGHT=0.0
export PDM_LATERAL_LOSS_WEIGHT=0.0
export PDM_CONTROLLER_LOSS_WEIGHT=0.0
export PDM_PLAN_STEER_LOSS_WEIGHT=0.0
export PDM_PLAN_THROTTLE_LOSS_WEIGHT=0.0
export PDM_PLAN_BRAKE_LOSS_WEIGHT=0.0

export SELECTION_METRIC=loss
export LR=${LR:-3.0e-5}
export BATCH_SIZE=${BATCH_SIZE:-24}
export NUM_WORKERS=${NUM_WORKERS:-4}
export EPOCHS=${EPOCHS:-10}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}

exec "$(dirname "$0")/train_exact_target_only_dl2.sh"
