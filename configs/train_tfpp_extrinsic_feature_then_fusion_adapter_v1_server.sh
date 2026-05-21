#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Extrinsic-aware Feature+Fusion Adapter v1.
#
# This is a fine-tune of Feature+Fusion Adapter v1, not a new output adapter.
# The pretrained feature+fusion adapter is loaded first, then zero-initialized
# extrinsic-conditioned gates are trained on top of each stage/fused adapter.
# With a single shifted rig this should be read as a geometry-prior ablation,
# not as full multi-rig generalization.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}
export PREV_WORK_ROOT=${PREV_WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_feature_then_fusion_adapter_v1"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_extrinsic_feature_then_fusion_adapter_v1"}
export VIEW_ROOT=${VIEW_ROOT:-"$PREV_WORK_ROOT/profile_views"}
export INDEX_DIR=${INDEX_DIR:-"$PREV_WORK_ROOT/indexes"}
export FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-"$PREV_WORK_ROOT/feature_fusion_cache"}
export TFPP_FEATURE_CACHE=${TFPP_FEATURE_CACHE:-"$FEATURE_CACHE_DIR/tfpp_ego_feature_fusion"}
export SHIFT_FEATURE_CACHE=${SHIFT_FEATURE_CACHE:-"$FEATURE_CACHE_DIR/front_triplet_shifted_feature_fusion"}
export OUT=${OUT:-"$WORK_ROOT/train_extrinsic_feature_then_fusion_adapter_v1"}
export INIT_CHECKPOINT=${INIT_CHECKPOINT:-"$PREV_WORK_ROOT/train_feature_then_fusion_adapter_v1/best_model.pt"}

export EXTRINSIC_AWARE=${EXTRINSIC_AWARE:-1}
export SOURCE_PROFILE=${SOURCE_PROFILE:-front_triplet_shifted}
export EXTRINSIC_HIDDEN_DIM=${EXTRINSIC_HIDDEN_DIM:-64}
export EXTRINSIC_DROPOUT=${EXTRINSIC_DROPOUT:-0.0}
export FREEZE_BASE=${FREEZE_BASE:-0}

export EPOCHS=${EPOCHS:-35}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-8}
export BATCH_SIZE=${BATCH_SIZE:-128}
export NUM_WORKERS=${NUM_WORKERS:-4}
export LR=${LR:-5e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export DATA_PARALLEL=${DATA_PARALLEL:-1}

# Reuse the existing feature-fusion caches by default.
export OVERWRITE=${OVERWRITE:-0}
export EXPORT_OVERWRITE=${EXPORT_OVERWRITE:-0}
export INDEX_OVERWRITE=${INDEX_OVERWRITE:-0}

# Keep close to the pretrained feature+fusion adapter while allowing the
# extrinsic gates to improve geometry-conditioned alignment.
export STAGE_RESIDUAL_LOSS_WEIGHT=${STAGE_RESIDUAL_LOSS_WEIGHT:-0.02}
export FUSED_RESIDUAL_LOSS_WEIGHT=${FUSED_RESIDUAL_LOSS_WEIGHT:-0.02}
export STAGE_COSINE_LOSS_WEIGHT=${STAGE_COSINE_LOSS_WEIGHT:-0.05}
export FUSED_COSINE_LOSS_WEIGHT=${FUSED_COSINE_LOSS_WEIGHT:-0.05}

exec bash configs/train_tfpp_feature_then_fusion_adapter_v1_server.sh
