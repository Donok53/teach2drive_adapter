#!/usr/bin/env bash
set -euo pipefail

# Stage 2: shifted RGB -> camera-stage adapters -> frozen TF++ + frozen stage-1
# turn/vehicle output adapter.  A canonical-camera copy of stage 1 supplies
# feature and full 10-checkpoint teacher targets.  LiDAR is passed through
# exactly, and no expert temporal/control objective is used.

: "${INIT_CHECKPOINT:?Set INIT_CHECKPOINT to stage-1 best_model.pt}"
export RUN_NAME=${RUN_NAME:-train_exact_stage2_camera_from_turn_vehicle_gpu1}
export TEACHER_VIEW_AS_INPUT=0
export FREEZE_INIT_AS_TEACHER=1
export TEACHER_DISTILL_ONLY=1

export STAGE_ADAPTER_LAYERS=${STAGE_ADAPTER_LAYERS:-all}
export STAGE_ADAPTER_MODALITIES=camera_keep_lidar
export DISABLE_FUSION_ADAPTER=1
export FREEZE_ADAPTER_INCLUDE='^adapter\.stage_adapters\.layer_[0-9]+_lidar\.,^adapter\.fused_adapter\.'
export EXTRINSIC_AWARE=0
export SOURCE_PROFILE=front_triplet_shifted
export LORA_RANK=0
export LORA_ALPHA=1
export LORA_INCLUDE=''
export UNFREEZE_INCLUDE=''

# Restore the stage-1 correction in both branches, but never optimize it here.
export OUTPUT_RESIDUAL=1
export OUTPUT_RESIDUAL_CHECKPOINT_LATERAL_ONLY=1
export FREEZE_OUTPUT_RESIDUAL=1
export OUTPUT_RESIDUAL_HIDDEN_DIM=256
export OUTPUT_RESIDUAL_CHECKPOINT_SCALE=0.75
export OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE=0.0
export OUTPUT_RESIDUAL_GATE_BIAS=-2.3
export OUTPUT_RESIDUAL_DROPOUT=0.0

# Student camera features and controller-facing outputs imitate the canonical
# stage-1 teacher.  The speed term prevents camera adaptation from changing the
# pretrained TF++ speed decision even though the stage-1 speed residual is zero.
export FEATURE_DRIFT_WEIGHT=${FEATURE_DRIFT_WEIGHT:-0.50}
export TEACHER_CHECKPOINT_LOSS_WEIGHT=${TEACHER_CHECKPOINT_LOSS_WEIGHT:-1.0}
export OUTPUT_PRIOR_SPEED_LOSS_WEIGHT=${OUTPUT_PRIOR_SPEED_LOSS_WEIGHT:-0.25}
export OUTPUT_PRIOR_XY_LOSS_WEIGHT=0.0

export CONTROL_LOSS_WEIGHT=0.0
export STOP_STATE_AUX_LOSS_WEIGHT=0.0
export STOP_REASON_AUX_LOSS_WEIGHT=0.0
export PDM_BEHAVIOR_LOSS_WEIGHT=0.0
export PDM_LATERAL_LOSS_WEIGHT=0.0
export PDM_CONTROLLER_LOSS_WEIGHT=0.0
export PDM_PLAN_STEER_LOSS_WEIGHT=0.0
export PDM_PLAN_THROTTLE_LOSS_WEIGHT=0.0
export PDM_PLAN_BRAKE_LOSS_WEIGHT=0.0

export SELECTION_METRIC=${SELECTION_METRIC:-loss}
export LR=${LR:-8.0e-6}
export BATCH_SIZE=${BATCH_SIZE:-16}
export NUM_WORKERS=${NUM_WORKERS:-4}
export EPOCHS=${EPOCHS:-10}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}

exec "$(dirname "$0")/train_exact_target_only_dl2.sh"
