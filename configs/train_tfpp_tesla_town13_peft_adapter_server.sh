#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Target-only PEFT run:
# - same target-only sensor supervision as the task feature adapter
# - plus small LoRA adapters on TF++ join/checkpoint/speed planner modules
# - no paired canonical sensor target

export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_tesla_town13_peft_adapter"}
export OUT=${OUT:-"$WORK_ROOT/train_front_triplet_shifted_feature_plus_planner_lora"}

export EPOCHS=${EPOCHS:-20}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-6}
export BATCH_SIZE=${BATCH_SIZE:-6}
export LR=${LR:-1e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}

export LORA_RANK=${LORA_RANK:-8}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export LORA_DROPOUT=${LORA_DROPOUT:-0.02}
export LORA_INCLUDE=${LORA_INCLUDE:-"^join\\.,^checkpoint_decoder\\.(encoder|decoder)\\.,^target_speed_network\\."}
export LORA_EXCLUDE=${LORA_EXCLUDE:-""}

# Keep feature changes small while planner LoRA learns speed/stop timing.
export FEATURE_DRIFT_LOSS_WEIGHT=${FEATURE_DRIFT_LOSS_WEIGHT:-0.16}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.10}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-2.5}
export STOP_SPEED_CEILING_LOSS_WEIGHT=${STOP_SPEED_CEILING_LOSS_WEIGHT:-0.45}

exec bash configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh
