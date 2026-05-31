#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Expert-only adapter plus auxiliary expert control supervision.
# The control head is training-only: it shapes adapted features using
# [steer, throttle, brake] labels, while evaluation still uses TF++ control.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_town13_front_triplet_snapshot_peft_dp2"}
export SNAPSHOT_COMPLETE_EPISODES=${SNAPSHOT_COMPLETE_EPISODES:-0}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_tesla_town13_expert_control_aux_lora8_blend1"}
export OUT=${OUT:-"$WORK_ROOT/train_front_triplet_shifted_expert_control_aux_lora8_blend1"}

export REFRESH_SNAPSHOT=${REFRESH_SNAPSHOT:-0}
export EXPORT_OVERWRITE=${EXPORT_OVERWRITE:-0}
export INDEX_OVERWRITE=${INDEX_OVERWRITE:-0}
export OVERWRITE=${OVERWRITE:-1}

export STAGE_FEATURE_ADAPTER_BLEND=${STAGE_FEATURE_ADAPTER_BLEND:-1.0}
export FUSION_ADAPTER_BLEND=${FUSION_ADAPTER_BLEND:-1.0}
export FEATURE_DRIFT_LOSS_WEIGHT=${FEATURE_DRIFT_LOSS_WEIGHT:-0.0}
export OUTPUT_PRIOR_XY_LOSS_WEIGHT=${OUTPUT_PRIOR_XY_LOSS_WEIGHT:-0.0}
export OUTPUT_PRIOR_SPEED_LOSS_WEIGHT=${OUTPUT_PRIOR_SPEED_LOSS_WEIGHT:-0.0}
export TRAJ_SMOOTH_LOSS_WEIGHT=${TRAJ_SMOOTH_LOSS_WEIGHT:-0.0}
export SPEED_SMOOTH_LOSS_WEIGHT=${SPEED_SMOOTH_LOSS_WEIGHT:-0.0}

export LORA_RANK=${LORA_RANK:-8}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export LORA_DROPOUT=${LORA_DROPOUT:-0.02}
export LR=${LR:-1e-5}
export BATCH_SIZE=${BATCH_SIZE:-8}
export EPOCHS=${EPOCHS:-24}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-8}

export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.10}
export CONTROL_LOSS_WEIGHT=${CONTROL_LOSS_WEIGHT:-0.60}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-2.5}
export STOP_SPEED_CEILING_LOSS_WEIGHT=${STOP_SPEED_CEILING_LOSS_WEIGHT:-0.45}

exec bash configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh
