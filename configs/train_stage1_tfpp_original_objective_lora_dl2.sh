#!/usr/bin/env bash
set -euo pipefail

# Capacity ablation for the exact TF++ objective.  Instead of an output-only
# residual, train LoRA modules in the original checkpoint decoder and target-
# speed head while keeping perception/fusion frozen.
export RUN_NAME=${RUN_NAME:-train_stage1_tfpp_original_objective_lora_v2_fixed_gpu0}
export INDEX=${INDEX:-/data/dataset/byeongjae/datasets/t2d_paired_shifted_3h_tfpp_exact_index.npz}
export EPISODE_ROOT=${EPISODE_ROOT:-/data/dataset/byeongjae/datasets/t2d_paired_shifted_3h_tfpp_exact}
export MEASUREMENT_ROOT=${MEASUREMENT_ROOT:-/data/dataset/byeongjae/datasets/pdm_lite_tesla_paired_3h/data}
export TEACHER_VIEW_ROOT=${TEACHER_VIEW_ROOT:-/data/dataset/byeongjae/datasets/pdm_lite_tesla_paired_3h/data}
export TEACHER_VIEW_DIRNAME=rgb_canonical
export TEACHER_VIEW_AS_INPUT=1

export STAGE_ADAPTER_MODALITIES=none
export DISABLE_FUSION_ADAPTER=1
export FREEZE_ADAPTER_INCLUDE='.*'
export EXTRINSIC_AWARE=0
export SOURCE_PROFILE=tfpp_ego
export FEATURE_DRIFT_WEIGHT=0.0
export OUTPUT_RESIDUAL=0
export LORA_RANK=${LORA_RANK:-8}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export LORA_INCLUDE='^checkpoint_decoder\.(encoder|decoder)$,^target_speed_network\.'
export LORA_REQUIRE_MODULES='checkpoint_decoder.encoder,checkpoint_decoder.decoder,target_speed_network.0,target_speed_network.2'
export UNFREEZE_INCLUDE=''

export ROUTE_TARGET_SOURCE=measurement_route
export ROUTE_TARGET_ONLY=0
export TFPP_ORIGINAL_OBJECTIVE=1
export TFPP_CHECKPOINT_LOSS_WEIGHT=1.0
export TFPP_TARGET_SPEED_LOSS_WEIGHT=1.0
export CONTROL_LOSS_WEIGHT=0.0
export STOP_STATE_AUX_LOSS_WEIGHT=0.0
export STOP_REASON_AUX_LOSS_WEIGHT=0.0

export SELECTION_METRIC=tfpp_original_loss
export LR=${LR:-1.0e-4}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1.0e-4}
export BATCH_SIZE=${BATCH_SIZE:-18}
export NUM_WORKERS=${NUM_WORKERS:-6}
export EPOCHS=${EPOCHS:-31}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-6}

exec "$(dirname "$0")/train_exact_target_only_dl2.sh"
