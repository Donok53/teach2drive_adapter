#!/usr/bin/env bash
set -euo pipefail

# Stage 1B ablation: preserve checkpoint planning and learn only a bounded
# target-speed-logit residual.  This is isolated from Stage 1A so the oracle
# ablation can be compared with like-for-like learned heads.

export RUN_NAME=${RUN_NAME:-train_stage1_policy_preserving_speed_gpu1}
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
export OUTPUT_RESIDUAL_CHECKPOINT_SCALE=0.0
export OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE=${OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE:-0.50}
export OUTPUT_RESIDUAL_GATE_BIAS=${OUTPUT_RESIDUAL_GATE_BIAS:--3.0}
export OUTPUT_RESIDUAL_DROPOUT=0.0

export ROUTE_TARGET_ONLY=0
export OUTPUT_PRIOR_XY_LOSS_WEIGHT=0.0
export OUTPUT_PRIOR_SPEED_LOSS_WEIGHT=${OUTPUT_PRIOR_SPEED_LOSS_WEIGHT:-1.0}
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
