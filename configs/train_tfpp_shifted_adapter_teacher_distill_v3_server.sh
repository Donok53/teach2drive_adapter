#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# v3 preset: canonical-teacher distillation.
# The adapter sees the same shifted TransFuser++ prior it will see at inference,
# then learns to recover the canonical TransFuser++ behavior from paired data.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}

export PREV_WORK_ROOT=${PREV_WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_shifted_adapter_motion_safe"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_shifted_adapter_motion_safe_e2e_v3_teacher_distill"}
export VIEW_ROOT=${VIEW_ROOT:-"$PREV_WORK_ROOT/profile_views"}
export INDEX_DIR=${INDEX_DIR:-"$PREV_WORK_ROOT/indexes"}

# Canonical cache is the teacher. Shifted cache is the input prior.
export CACHE=${CACHE:-"$PREV_WORK_ROOT/cache/tfpp_ego_prior_cache.npz"}
export SHIFT_CACHE=${SHIFT_CACHE:-"$WORK_ROOT/cache/front_triplet_shifted_prior_cache.npz"}
export TRAIN_CACHE=${TRAIN_CACHE:-"$SHIFT_CACHE"}
export TEACHER_CACHE=${TEACHER_CACHE:-"$CACHE"}
export BUILD_SHIFT_CACHE=${BUILD_SHIFT_CACHE:-1}
export OUT=${OUT:-"$WORK_ROOT/train_shifted_visual_layout_adapter_teacher_distill"}

export BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-"$HOME/teach2drive/workspace/teach2drive_bootstrap"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
export TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}

export EPOCHS=${EPOCHS:-40}
export BATCH_SIZE=${BATCH_SIZE:-128}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-64}
export NUM_WORKERS=${NUM_WORKERS:-8}
export CACHE_WORKERS=${CACHE_WORKERS:-16}
export LR=${LR:-1.25e-4}
export VAL_RATIO=${VAL_RATIO:-0.15}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export SKIP_INVALID_MOTION=${SKIP_INVALID_MOTION:-1}

export MOVING_SPEED_THRESHOLD=${MOVING_SPEED_THRESHOLD:-1.0}
export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-2.0}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-0.50}

# Mostly imitate the canonical TransFuser++ output, while retaining a small
# amount of the collected expert labels so the model stays anchored to the
# route/data distribution.
export TEACHER_TARGET_BLEND=${TEACHER_TARGET_BLEND:-0.75}
export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-1.0}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.05}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.45}

# The input prior is shifted-sensor TransFuser++; keep it only as a weak
# regularizer so the adapter learns a correction, not a brand-new policy.
export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.05}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.02}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.08}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-1.0}
export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.01}

# No temporal/smoothing losses in this preset; keep the improvement attributable
# to sensor-to-canonical distillation.
export SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.0}
export SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.0}
export TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.0}
export TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.0}

export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.005}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.01}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.0}
export STOP_POSITIVE_LOSS_SCALE=${STOP_POSITIVE_LOSS_SCALE:-0.25}
export STOP_NEGATIVE_LOSS_SCALE=${STOP_NEGATIVE_LOSS_SCALE:-2.0}
export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-4}

exec bash configs/train_tfpp_shifted_adapter_server.sh
