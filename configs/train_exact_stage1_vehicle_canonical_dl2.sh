#!/usr/bin/env bash
set -euo pipefail

# Stage 1: learn Tesla vehicle behavior from the canonical paired camera.
# The paired dataset has only one LiDAR, so the identical shifted LiDAR is used
# here and in stage 2. No sensor-distillation loss is used in this stage.

export RUN_NAME=${RUN_NAME:-train_exact_stage1_vehicle_canonical_gpu0}
export TEACHER_VIEW_AS_INPUT=1
export FEATURE_DRIFT_WEIGHT=0.0
export STAGE_ADAPTER_LAYERS=${STAGE_ADAPTER_LAYERS:-all}
export STAGE_ADAPTER_MODALITIES=${STAGE_ADAPTER_MODALITIES:-all}
export DISABLE_FUSION_ADAPTER=0
export EXTRINSIC_AWARE=0
export SOURCE_PROFILE=tfpp_ego
export LORA_RANK=0
export LORA_ALPHA=1
export LORA_INCLUDE=''
export UNFREEZE_INCLUDE=''

exec "$(dirname "$0")/train_exact_target_only_dl2.sh"
