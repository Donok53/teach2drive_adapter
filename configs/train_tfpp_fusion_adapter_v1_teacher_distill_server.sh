#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Fusion Adapter v1.
#
# This is the same deployment insertion point as FA-v1: the fused camera/LiDAR
# feature map returned by the frozen TransFuser++ backbone. The experiment is
# named as "fusion" explicitly because, in the user's taxonomy, the adapter is
# applied after camera and LiDAR have already been fused.
#
# Objective:
#   shifted fused feature -> residual Conv adapter -> canonical TF++ fused feature
#
# It is a representation-only teacher-distillation baseline. The checkpoint is
# compatible with scripts/tfpp_feature_adapter_sensor_rig_agent.py.

export WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_fusion_adapter_v1_teacher_distill"}
export OUT=${OUT:-"$WORK_ROOT/train_fusion_adapter_v1_teacher_distill"}

export FEATURE_LOSS_WEIGHT=${FEATURE_LOSS_WEIGHT:-1.0}
export COSINE_LOSS_WEIGHT=${COSINE_LOSS_WEIGHT:-0.08}
export RESIDUAL_LOSS_WEIGHT=${RESIDUAL_LOSS_WEIGHT:-0.01}
export BLOCKS=${BLOCKS:-3}
export DROPOUT=${DROPOUT:-0.02}
export LR=${LR:-2e-4}
export EPOCHS=${EPOCHS:-60}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-12}

bash configs/train_tfpp_feature_adapter_v1_fused_server.sh
