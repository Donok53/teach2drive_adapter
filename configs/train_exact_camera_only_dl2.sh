#!/usr/bin/env bash
set -euo pipefail

# Paired camera-only adaptation:
#   teacher = canonical RGB + the same shifted LiDAR/scalars
#   student = shifted RGB   + the same shifted LiDAR/scalars
# Only camera-stage adapters are inserted. LiDAR-stage adapters, the fused
# adapter, TF++ LoRA, and base-network unfreezing are disabled.

export RUN_NAME=${RUN_NAME:-train_exact_camera_only}
export FEATURE_DRIFT_WEIGHT=${FEATURE_DRIFT_WEIGHT:-0.18}
export STAGE_ADAPTER_LAYERS=${STAGE_ADAPTER_LAYERS:-all}
export STAGE_ADAPTER_MODALITIES=camera
export DISABLE_FUSION_ADAPTER=1
export LORA_RANK=0
export LORA_ALPHA=1
export LORA_INCLUDE=''
export UNFREEZE_INCLUDE=''

exec "$(dirname "$0")/train_exact_target_only_dl2.sh"
