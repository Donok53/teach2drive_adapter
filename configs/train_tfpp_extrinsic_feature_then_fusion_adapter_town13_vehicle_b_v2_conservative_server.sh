#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Target-domain conservative fine-tune of the current best feature/fusion adapter.
#
# This intentionally keeps the v2 feature/fusion adapter structure:
#   - source: front_triplet_shifted feature/fusion cache
#   - target: tfpp_ego feature/fusion cache
#   - init: global best v2 conservative checkpoint
#   - frozen base adapters + low-LR extrinsic-aware gate refinement
#
# It does not train an output residual, policy-recovery head, or full TF++ model.

export PY=${PY:-python}

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_vehicle_b_town13_paired_tfpp_ego_front_triplet_3h"}
export BASE_WORK_ROOT=${BASE_WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_feature_then_fusion_adapter_town13_vehicle_b_cache"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_extrinsic_feature_then_fusion_adapter_town13_vehicle_b_v2_conservative"}
export INIT_CHECKPOINT=${INIT_CHECKPOINT:-"$HOME/dataset/byeongjae/runs/tfpp_extrinsic_feature_then_fusion_adapter_v2_conservative/train_extrinsic_feature_then_fusion_adapter_v2_conservative/best_model.pt"}

export DATA_PARALLEL=${DATA_PARALLEL:-1}
export BATCH_SIZE=${BATCH_SIZE:-128}
export EPOCHS=${EPOCHS:-20}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-6}
export LR=${LR:-1e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export NUM_WORKERS=${NUM_WORKERS:-4}

export OVERWRITE=${OVERWRITE:-0}
export EXPORT_OVERWRITE=${EXPORT_OVERWRITE:-0}
export INDEX_OVERWRITE=${INDEX_OVERWRITE:-0}
export FEATURE_CACHE_BATCH_SIZE=${FEATURE_CACHE_BATCH_SIZE:-32}
export FEATURE_CACHE_WORKERS=${FEATURE_CACHE_WORKERS:-8}

echo "=== validate Town13 paired dataset"
"$PY" scripts/validate_paired_profile_dataset.py "$DATA_ROOT"

exec bash configs/train_tfpp_extrinsic_feature_then_fusion_adapter_v2_conservative_server.sh
