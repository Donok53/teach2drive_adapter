#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# v3.1 preset: conservative canonical-teacher distillation.
# v3 recovered route following, but was too aggressive at junctions/signals.
# This preset reuses the shifted prior cache from v3, lowers the canonical
# teacher blend, and trusts expert speed/slowdown labels more.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}

export PREV_WORK_ROOT=${PREV_WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_shifted_adapter_motion_safe"}
export V3_WORK_ROOT=${V3_WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_shifted_adapter_motion_safe_e2e_v3_teacher_distill"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_shifted_adapter_motion_safe_e2e_v31_teacher_distill_conservative"}
export VIEW_ROOT=${VIEW_ROOT:-"$PREV_WORK_ROOT/profile_views"}
export INDEX_DIR=${INDEX_DIR:-"$PREV_WORK_ROOT/indexes"}

export CACHE=${CACHE:-"$PREV_WORK_ROOT/cache/tfpp_ego_prior_cache.npz"}
export SHIFT_CACHE=${SHIFT_CACHE:-"$V3_WORK_ROOT/cache/front_triplet_shifted_prior_cache.npz"}
export TRAIN_CACHE=${TRAIN_CACHE:-"$SHIFT_CACHE"}
export TEACHER_CACHE=${TEACHER_CACHE:-"$CACHE"}
export BUILD_SHIFT_CACHE=${BUILD_SHIFT_CACHE:-1}
export OUT=${OUT:-"$WORK_ROOT/train_shifted_visual_layout_adapter_teacher_distill_v31"}

export BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-"$HOME/teach2drive/workspace/teach2drive_bootstrap"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
export TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}

export EPOCHS=${EPOCHS:-40}
export BATCH_SIZE=${BATCH_SIZE:-128}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
export NUM_WORKERS=${NUM_WORKERS:-8}
export CACHE_WORKERS=${CACHE_WORKERS:-24}
export LR=${LR:-1.25e-4}
export VAL_RATIO=${VAL_RATIO:-0.15}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export SKIP_INVALID_MOTION=${SKIP_INVALID_MOTION:-1}

# V3 was route-capable but too fast/aggressive. Keep enough moving weight to
# avoid collapse, but give slow/stopped frames more influence than v3.
export MOVING_SPEED_THRESHOLD=${MOVING_SPEED_THRESHOLD:-1.0}
export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.5}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-0.8}

# Lower than v3's 0.75. This keeps canonical route intent but pulls speed and
# cautious junction behavior back toward the collected expert labels.
export TEACHER_TARGET_BLEND=${TEACHER_TARGET_BLEND:-0.50}
export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-1.0}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.05}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.75}

# Do not distill speed toward the shifted input prior; it was also fast.
export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.0}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.0}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.02}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-0.5}
export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.0}

export SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.0}
export SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.0}
export TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.0}
export TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.0}

export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.02}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.04}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.0}
export STOP_POSITIVE_LOSS_SCALE=${STOP_POSITIVE_LOSS_SCALE:-0.5}
export STOP_NEGATIVE_LOSS_SCALE=${STOP_NEGATIVE_LOSS_SCALE:-1.5}
export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-3}

exec bash configs/train_tfpp_shifted_adapter_server.sh
