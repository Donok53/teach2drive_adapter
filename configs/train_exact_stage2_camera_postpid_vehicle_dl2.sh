#!/usr/bin/env bash
set -euo pipefail

# Stage 2 for the redesigned post-PID vehicle adapter. The vehicle adapter is
# trained and deployed separately, so this stage learns only the sensor-view
# mapping: shifted RGB features -> canonical-camera TF++ features/outputs.
export RUN_NAME=${RUN_NAME:-train_exact_stage2_camera_postpid_vehicle}
export INDEX=${INDEX:-/data/dataset/byeongjae/datasets/t2d_paired_shifted_3h_tfpp_exact_index.npz}
export EPISODE_ROOT=${EPISODE_ROOT:-/data/dataset/byeongjae/datasets/t2d_paired_shifted_3h_tfpp_exact}
export TEACHER_VIEW_ROOT=${TEACHER_VIEW_ROOT:-/data/dataset/byeongjae/datasets/pdm_lite_tesla_paired_3h/data}
export TEACHER_VIEW_DIRNAME=${TEACHER_VIEW_DIRNAME:-rgb_canonical}
export TEACHER_VIEW_AS_INPUT=0
export TEACHER_DISTILL_ONLY=1

export STAGE_ADAPTER_LAYERS=${STAGE_ADAPTER_LAYERS:-all}
export STAGE_ADAPTER_MODALITIES=camera_keep_lidar
export DISABLE_FUSION_ADAPTER=1
export EXTRINSIC_AWARE=${EXTRINSIC_AWARE:-0}
export SOURCE_PROFILE=front_triplet_shifted
export LORA_RANK=0
export LORA_ALPHA=1
export LORA_INCLUDE=''
export UNFREEZE_INCLUDE=''

export OUTPUT_RESIDUAL=0
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
