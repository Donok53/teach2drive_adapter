#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# This preset is for the "shifted sensor rig, learned control" experiment.
# It reuses the expensive TransFuser++ prior cache when available, but trains a
# new adapter with stronger moving-frame and speed losses so the policy does not
# collapse into standing still.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}

export PREV_WORK_ROOT=${PREV_WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_shifted_adapter_motion_safe"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_shifted_adapter_motion_safe_e2e_v2"}
export VIEW_ROOT=${VIEW_ROOT:-"$PREV_WORK_ROOT/profile_views"}
export INDEX_DIR=${INDEX_DIR:-"$PREV_WORK_ROOT/indexes"}
export CACHE=${CACHE:-"$PREV_WORK_ROOT/cache/tfpp_ego_prior_cache.npz"}
export OUT=${OUT:-"$WORK_ROOT/train_shifted_visual_layout_adapter_balanced_e2e"}

export BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-"$HOME/teach2drive/workspace/teach2drive_bootstrap"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
export TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}

export EPOCHS=${EPOCHS:-12}
export BATCH_SIZE=${BATCH_SIZE:-128}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-64}
export NUM_WORKERS=${NUM_WORKERS:-8}
export CACHE_WORKERS=${CACHE_WORKERS:-16}
export LR=${LR:-1.5e-4}
export VAL_RATIO=${VAL_RATIO:-0.15}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export SKIP_INVALID_MOTION=${SKIP_INVALID_MOTION:-1}

# Keep the trajectory head primary, but make learned speed hard to ignore.
export MOVING_SPEED_THRESHOLD=${MOVING_SPEED_THRESHOLD:-1.0}
export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-2.5}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-0.35}
export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-1.0}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.05}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.60}
export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.30}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.15}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.35}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-2.0}

# Stop labels are useful metadata, but should not dominate this short-data
# adaptation run. Delay and down-weight them to avoid a "never move" policy.
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.005}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.01}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.0}
export STOP_POSITIVE_LOSS_SCALE=${STOP_POSITIVE_LOSS_SCALE:-0.25}
export STOP_NEGATIVE_LOSS_SCALE=${STOP_NEGATIVE_LOSS_SCALE:-2.0}
export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-4}

exec bash configs/train_tfpp_shifted_adapter_server.sh
