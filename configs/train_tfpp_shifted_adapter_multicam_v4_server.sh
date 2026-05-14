#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# v4 preset: traffic-label-restored, multi-camera-heavy adapter.
# This keeps the TransFuser++ prior/teacher setup from v3.3, but gives the
# visual adapter more capacity and occasionally drops the front camera during
# training so left/right views become useful instead of decorative.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}

export PREV_WORK_ROOT=${PREV_WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_shifted_adapter_motion_safe"}
export V3_WORK_ROOT=${V3_WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_shifted_adapter_motion_safe_e2e_v3_teacher_distill"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_shifted_adapter_motion_safe_e2e_v4_multicam"}
export VIEW_ROOT=${VIEW_ROOT:-"$WORK_ROOT/profile_views"}
export INDEX_DIR=${INDEX_DIR:-"$WORK_ROOT/indexes"}

export CACHE=${CACHE:-"$PREV_WORK_ROOT/cache/tfpp_ego_prior_cache.npz"}
export SHIFT_CACHE=${SHIFT_CACHE:-"$V3_WORK_ROOT/cache/front_triplet_shifted_prior_cache.npz"}
export TRAIN_CACHE=${TRAIN_CACHE:-"$SHIFT_CACHE"}
export TEACHER_CACHE=${TEACHER_CACHE:-"$CACHE"}
export BUILD_SHIFT_CACHE=${BUILD_SHIFT_CACHE:-1}
export OUT=${OUT:-"$WORK_ROOT/train_shifted_visual_layout_adapter_multicam_v4"}

export EXPORT_OVERWRITE=${EXPORT_OVERWRITE:-1}
export INDEX_OVERWRITE=${INDEX_OVERWRITE:-1}
export OVERWRITE=${OVERWRITE:-0}

export BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-"$HOME/teach2drive/workspace/teach2drive_bootstrap"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
export TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}

export EPOCHS=${EPOCHS:-60}
export BATCH_SIZE=${BATCH_SIZE:-96}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
export NUM_WORKERS=${NUM_WORKERS:-8}
export CACHE_WORKERS=${CACHE_WORKERS:-24}
export LR=${LR:-8e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export VAL_RATIO=${VAL_RATIO:-0.15}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export SKIP_INVALID_MOTION=${SKIP_INVALID_MOTION:-1}

# More visual capacity than v3.x. Batch is slightly lower to avoid memory spikes.
export HIDDEN_DIM=${HIDDEN_DIM:-768}
export LAYOUT_HIDDEN_DIM=${LAYOUT_HIDDEN_DIM:-192}
export VISUAL_DIM=${VISUAL_DIM:-384}
export VISUAL_TOKEN_DIM=${VISUAL_TOKEN_DIM:-256}
export VISUAL_LAYERS=${VISUAL_LAYERS:-4}
export VISUAL_HEADS=${VISUAL_HEADS:-8}

# Encourage the adapter to use side views. This is train-time only; inference
# still uses all left/front/right cameras.
export CAMERA_DROPOUT_PROB=${CAMERA_DROPOUT_PROB:-0.02}
export FRONT_CAMERA_DROPOUT_PROB=${FRONT_CAMERA_DROPOUT_PROB:-0.20}

export MOVING_SPEED_THRESHOLD=${MOVING_SPEED_THRESHOLD:-1.0}
export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.25}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-1.0}

export TEACHER_TARGET_BLEND=${TEACHER_TARGET_BLEND:-0.0}
export TEACHER_TRAJ_BLEND=${TEACHER_TRAJ_BLEND:-0.65}
export TEACHER_SPEED_TARGET_BLEND=${TEACHER_SPEED_TARGET_BLEND:-0.0}
export TEACHER_STOP_TARGET_BLEND=${TEACHER_STOP_TARGET_BLEND:-0.0}

export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-1.0}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.05}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-1.0}

export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.0}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.0}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.01}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-0.3}
export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.0}

export SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.08}
export SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.03}
export TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.03}
export TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.01}

export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.05}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.10}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.04}
export STOP_POSITIVE_LOSS_SCALE=${STOP_POSITIVE_LOSS_SCALE:-1.0}
export STOP_NEGATIVE_LOSS_SCALE=${STOP_NEGATIVE_LOSS_SCALE:-1.0}
export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-1}

exec bash configs/train_tfpp_shifted_adapter_server.sh
