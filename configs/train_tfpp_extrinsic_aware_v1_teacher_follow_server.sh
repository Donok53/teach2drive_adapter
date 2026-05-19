#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Extrinsic-aware v1.
#
# This is a new experiment family, separate from Vehicle-B output-adapter v1-v7.
# Goal: learn a layout/extrinsic-conditioned adapter that maps the shifted
# front_triplet sensor rig back toward the canonical TransFuser++ policy.
#
# Required data shape:
#   paired episodes containing both tfpp_ego and front_triplet_shifted profiles.
#
# Training signal:
#   input prior/cache  : front_triplet_shifted TransFuser++ prior
#   teacher prior/cache: canonical tfpp_ego TransFuser++ prior
#   raw sensor input   : front_triplet_shifted left/front/right + lidar
#   layout input       : front_triplet_shifted sensor extrinsics

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_extrinsic_aware_v1_teacher_follow"}
export VIEW_ROOT=${VIEW_ROOT:-"$WORK_ROOT/profile_views"}
export INDEX_DIR=${INDEX_DIR:-"$WORK_ROOT/indexes"}

export CACHE=${CACHE:-"$WORK_ROOT/cache/tfpp_ego_prior_cache.npz"}
export SHIFT_CACHE=${SHIFT_CACHE:-"$WORK_ROOT/cache/front_triplet_shifted_prior_cache.npz"}
export TRAIN_CACHE=${TRAIN_CACHE:-"$SHIFT_CACHE"}
export TEACHER_CACHE=${TEACHER_CACHE:-"$CACHE"}
export BUILD_SHIFT_CACHE=${BUILD_SHIFT_CACHE:-1}
export OUT=${OUT:-"$WORK_ROOT/train_extrinsic_aware_v1_teacher_follow"}

export BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-"$HOME/teach2drive/workspace/teach2drive_bootstrap"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
export TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}

export EXPORT_OVERWRITE=${EXPORT_OVERWRITE:-0}
export INDEX_OVERWRITE=${INDEX_OVERWRITE:-0}
export OVERWRITE=${OVERWRITE:-0}

export EPOCHS=${EPOCHS:-50}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-15}
export BATCH_SIZE=${BATCH_SIZE:-128}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
export NUM_WORKERS=${NUM_WORKERS:-8}
export CACHE_WORKERS=${CACHE_WORKERS:-24}
export LR=${LR:-8e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export VAL_RATIO=${VAL_RATIO:-0.15}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export SKIP_INVALID_MOTION=${SKIP_INVALID_MOTION:-1}
export AUGMENTATIONS=${AUGMENTATIONS:-0}

# Keep enough capacity for layout-conditioned visual adaptation.
export HIDDEN_DIM=${HIDDEN_DIM:-768}
export LAYOUT_HIDDEN_DIM=${LAYOUT_HIDDEN_DIM:-256}
export VISUAL_DIM=${VISUAL_DIM:-384}
export VISUAL_TOKEN_DIM=${VISUAL_TOKEN_DIM:-256}
export VISUAL_LAYERS=${VISUAL_LAYERS:-4}
export VISUAL_HEADS=${VISUAL_HEADS:-8}

# Encourage side cameras to matter during training.
export CAMERA_DROPOUT_PROB=${CAMERA_DROPOUT_PROB:-0.02}
export FRONT_CAMERA_DROPOUT_PROB=${FRONT_CAMERA_DROPOUT_PROB:-0.20}

# This experiment should follow the canonical policy first, not invent a new
# Vehicle-B driving style. Teacher blending is therefore stronger than in v1-v7.
export TEACHER_TARGET_BLEND=${TEACHER_TARGET_BLEND:-0.0}
export TEACHER_TRAJ_BLEND=${TEACHER_TRAJ_BLEND:-0.90}
export TEACHER_SPEED_TARGET_BLEND=${TEACHER_SPEED_TARGET_BLEND:-0.75}
export TEACHER_STOP_TARGET_BLEND=${TEACHER_STOP_TARGET_BLEND:-0.0}

export MOVING_SPEED_THRESHOLD=${MOVING_SPEED_THRESHOLD:-1.0}
export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.0}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-1.0}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-1.0}
export HAZARD_STOP_REASONS=${HAZARD_STOP_REASONS:-""}
export HAZARD_SPEED_TARGET_BLEND=${HAZARD_SPEED_TARGET_BLEND:--1.0}

export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-1.0}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.05}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.55}

# Do not regularize back toward the shifted zero-shot prior; that is the
# mismatched behavior this experiment is trying to correct.
export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.0}
export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.0}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.0}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.0}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-0.3}

export SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.02}
export SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.01}
export TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.01}
export TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.005}

# Stop labels are auxiliary here. The primary goal is sensor/extrinsic policy
# transfer, not a new handcrafted stop/go behavior policy.
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.005}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.02}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.05}
export STOP_POSITIVE_LOSS_SCALE=${STOP_POSITIVE_LOSS_SCALE:-0.75}
export STOP_NEGATIVE_LOSS_SCALE=${STOP_NEGATIVE_LOSS_SCALE:-1.25}
export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-6}

exec bash configs/train_tfpp_shifted_adapter_server.sh
