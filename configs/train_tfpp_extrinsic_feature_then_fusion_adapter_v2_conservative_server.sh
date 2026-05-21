#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Extrinsic-aware Feature+Fusion Adapter v2: conservative gate refinement.
#
# This keeps the output/policy adapter out of the loop.  It starts from the
# v1 extrinsic feature+fusion checkpoint, freezes the base feature/fusion
# adapters, and fine-tunes only the extrinsic-conditioned gates.  The extra
# base-consistency losses keep the gated features close to the frozen v1 base
# output so the adapter cannot over-correct as aggressively.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}
export BASE_WORK_ROOT=${BASE_WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_feature_then_fusion_adapter_v1"}
export V1_WORK_ROOT=${V1_WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_extrinsic_feature_then_fusion_adapter_v1"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_extrinsic_feature_then_fusion_adapter_v2_conservative"}

# Reuse the feature/fusion caches from the base feature+fusion run.
export PREV_WORK_ROOT=${PREV_WORK_ROOT:-"$BASE_WORK_ROOT"}
export VIEW_ROOT=${VIEW_ROOT:-"$BASE_WORK_ROOT/profile_views"}
export INDEX_DIR=${INDEX_DIR:-"$BASE_WORK_ROOT/indexes"}
export FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-"$BASE_WORK_ROOT/feature_fusion_cache"}
export TFPP_FEATURE_CACHE=${TFPP_FEATURE_CACHE:-"$FEATURE_CACHE_DIR/tfpp_ego_feature_fusion"}
export SHIFT_FEATURE_CACHE=${SHIFT_FEATURE_CACHE:-"$FEATURE_CACHE_DIR/front_triplet_shifted_feature_fusion"}

export OUT=${OUT:-"$WORK_ROOT/train_extrinsic_feature_then_fusion_adapter_v2_conservative"}
export INIT_CHECKPOINT=${INIT_CHECKPOINT:-"$V1_WORK_ROOT/train_extrinsic_feature_then_fusion_adapter_v1/best_model.pt"}

export EXTRINSIC_AWARE=${EXTRINSIC_AWARE:-1}
export SOURCE_PROFILE=${SOURCE_PROFILE:-front_triplet_shifted}
export EXTRINSIC_HIDDEN_DIM=${EXTRINSIC_HIDDEN_DIM:-64}
export EXTRINSIC_DROPOUT=${EXTRINSIC_DROPOUT:-0.0}
export FREEZE_BASE=${FREEZE_BASE:-1}

export EPOCHS=${EPOCHS:-30}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-8}
export BATCH_SIZE=${BATCH_SIZE:-128}
export NUM_WORKERS=${NUM_WORKERS:-4}
export LR=${LR:-2e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export DATA_PARALLEL=${DATA_PARALLEL:-1}

# Reuse cached profile/index/feature files unless explicitly overridden.
export OVERWRITE=${OVERWRITE:-0}
export EXPORT_OVERWRITE=${EXPORT_OVERWRITE:-0}
export INDEX_OVERWRITE=${INDEX_OVERWRITE:-0}

# Conservative alignment: still match canonical features, but penalize both
# total residual from the shifted feature and gate-only deviation from the
# frozen base adapter output.  Fused features get the stronger gate penalty
# because they sit closest to the TransFuser++ policy head.
export STAGE_FEATURE_LOSS_WEIGHT=${STAGE_FEATURE_LOSS_WEIGHT:-1.0}
export FUSED_FEATURE_LOSS_WEIGHT=${FUSED_FEATURE_LOSS_WEIGHT:-1.0}
export STAGE_COSINE_LOSS_WEIGHT=${STAGE_COSINE_LOSS_WEIGHT:-0.04}
export FUSED_COSINE_LOSS_WEIGHT=${FUSED_COSINE_LOSS_WEIGHT:-0.04}
export STAGE_RESIDUAL_LOSS_WEIGHT=${STAGE_RESIDUAL_LOSS_WEIGHT:-0.035}
export FUSED_RESIDUAL_LOSS_WEIGHT=${FUSED_RESIDUAL_LOSS_WEIGHT:-0.045}
export STAGE_BASE_CONSISTENCY_LOSS_WEIGHT=${STAGE_BASE_CONSISTENCY_LOSS_WEIGHT:-0.15}
export FUSED_BASE_CONSISTENCY_LOSS_WEIGHT=${FUSED_BASE_CONSISTENCY_LOSS_WEIGHT:-0.25}

exec bash configs/train_tfpp_feature_then_fusion_adapter_v1_server.sh
