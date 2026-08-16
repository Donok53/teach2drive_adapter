#!/usr/bin/env bash
set -euo pipefail

# Stage 1: vehicle-only adaptation on the paired canonical camera.
#
# TF++ perception, fusion and planner stay frozen.  Only the zero-initialized,
# bounded residual on controller-facing checkpoints and target-speed logits is
# trained.  Unlike the old auxiliary control head, this residual is restored and
# used by the closed-loop evaluation agent.
#
# The paired collection has no canonical LiDAR, so the same LiDAR input is used
# in both stages.  No LiDAR/fusion feature adapter is allowed to learn here.

export RUN_NAME=${RUN_NAME:-train_exact_stage1_vehicle_output_gpu1}
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
export OUTPUT_RESIDUAL_HIDDEN_DIM=${OUTPUT_RESIDUAL_HIDDEN_DIM:-256}
export OUTPUT_RESIDUAL_CHECKPOINT_SCALE=${OUTPUT_RESIDUAL_CHECKPOINT_SCALE:-0.60}
export OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE=${OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE:-1.25}
export OUTPUT_RESIDUAL_GATE_BIAS=${OUTPUT_RESIDUAL_GATE_BIAS:--2.3}
export OUTPUT_RESIDUAL_DROPOUT=0.0

# These auxiliary heads are not consumed by the deployed agent.  Keep them off
# so they cannot spend capacity or push gradients into perception features.
export CONTROL_LOSS_WEIGHT=0.0
export STOP_STATE_AUX_LOSS_WEIGHT=0.0
export STOP_REASON_AUX_LOSS_WEIGHT=0.0

# Preserve the pretrained TF++ plan as a soft anchor while permitting the small
# Tesla-specific correction represented by the bounded output head.
export OUTPUT_PRIOR_XY_LOSS_WEIGHT=${OUTPUT_PRIOR_XY_LOSS_WEIGHT:-0.30}
export OUTPUT_PRIOR_SPEED_LOSS_WEIGHT=${OUTPUT_PRIOR_SPEED_LOSS_WEIGHT:-0.20}
export PDM_BEHAVIOR_LOSS_WEIGHT=${PDM_BEHAVIOR_LOSS_WEIGHT:-0.04}
export PDM_LATERAL_LOSS_WEIGHT=${PDM_LATERAL_LOSS_WEIGHT:-0.08}
export PDM_CONTROLLER_LOSS_WEIGHT=${PDM_CONTROLLER_LOSS_WEIGHT:-0.03}
export PDM_PLAN_STEER_LOSS_WEIGHT=${PDM_PLAN_STEER_LOSS_WEIGHT:-0.05}
export PDM_PLAN_THROTTLE_LOSS_WEIGHT=${PDM_PLAN_THROTTLE_LOSS_WEIGHT:-0.02}
export PDM_PLAN_BRAKE_LOSS_WEIGHT=${PDM_PLAN_BRAKE_LOSS_WEIGHT:-0.03}

export SELECTION_METRIC=${SELECTION_METRIC:-controller_closed_loop_proxy}
export LR=${LR:-1.5e-5}
export BATCH_SIZE=${BATCH_SIZE:-24}
export NUM_WORKERS=${NUM_WORKERS:-4}
export EPOCHS=${EPOCHS:-12}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-4}

exec "$(dirname "$0")/train_exact_target_only_dl2.sh"
