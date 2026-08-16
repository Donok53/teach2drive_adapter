#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Target-only PEFT + expert BC + TF++ prior regularization, with temporal
# smoothness losses and a lower LR to reduce closed-loop steering/speed jitter.

export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_tesla_town13_peft_expert_bc_prior_smooth_sgpu"}
export OUT=${OUT:-"$WORK_ROOT/train_front_triplet_shifted_peft_expert_bc_prior_smooth_sgpu"}

export SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_town13_front_triplet_snapshot_peft_dp2"}
export REFRESH_SNAPSHOT=${REFRESH_SNAPSHOT:-0}
export EXPORT_OVERWRITE=${EXPORT_OVERWRITE:-0}
export INDEX_OVERWRITE=${INDEX_OVERWRITE:-0}
export OVERWRITE=${OVERWRITE:-1}

export EPOCHS=${EPOCHS:-20}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-6}
export BATCH_SIZE=${BATCH_SIZE:-8}
export LR=${LR:-5e-6}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export DATA_PARALLEL=${DATA_PARALLEL:-0}

export LORA_RANK=${LORA_RANK:-8}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export LORA_DROPOUT=${LORA_DROPOUT:-0.02}
export LORA_INCLUDE=${LORA_INCLUDE:-"^join\\.,^checkpoint_decoder\\.(encoder|decoder)\\.,^target_speed_network\\."}
export LORA_EXCLUDE=${LORA_EXCLUDE:-""}

export FEATURE_DRIFT_LOSS_WEIGHT=${FEATURE_DRIFT_LOSS_WEIGHT:-0.16}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.10}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-2.5}
export STOP_SPEED_CEILING_LOSS_WEIGHT=${STOP_SPEED_CEILING_LOSS_WEIGHT:-0.45}

export TRAJ_SMOOTH_LOSS_WEIGHT=${TRAJ_SMOOTH_LOSS_WEIGHT:-0.08}
export SPEED_SMOOTH_LOSS_WEIGHT=${SPEED_SMOOTH_LOSS_WEIGHT:-0.05}

exec bash configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh
