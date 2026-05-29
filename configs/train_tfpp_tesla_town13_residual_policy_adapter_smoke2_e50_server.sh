#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Second smoke run for the Tesla Town13 residual policy adapter.
# This wrapper keeps the launch command short on the remote server while using
# only complete collected episodes from the live dataset snapshot.

export SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_town13_paired_tfpp_ego_front_triplet_3h"}
export SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_town13_paired_tfpp_ego_front_triplet_train_snapshot_complete"}
export SNAPSHOT_COMPLETE_EPISODES=${SNAPSHOT_COMPLETE_EPISODES:-1}
export REFRESH_SNAPSHOT=${REFRESH_SNAPSHOT:-1}
export SNAPSHOT_MIN_FRAMES=${SNAPSHOT_MIN_FRAMES:-1150}
export SNAPSHOT_REQUIRED_PROFILES=${SNAPSHOT_REQUIRED_PROFILES:-"front_triplet_shifted"}

export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_tesla_town13_residual_policy_adapter_smoke2_complete"}
export OUT=${OUT:-"$WORK_ROOT/train_front_triplet_shifted_residual_policy_adapter_e50"}

export PROFILE=${PROFILE:-front_triplet_shifted}
export CAMERAS=${CAMERAS:-left,front,right}

export EPOCHS=${EPOCHS:-50}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-10}
export BATCH_SIZE=${BATCH_SIZE:-128}
export LR=${LR:-5e-5}

export DATA_PARALLEL=${DATA_PARALLEL:-1}
export CACHE_DATA_PARALLEL=${CACHE_DATA_PARALLEL:-$DATA_PARALLEL}
export TRAIN_DATA_PARALLEL=${TRAIN_DATA_PARALLEL:-$DATA_PARALLEL}

export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.10}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.05}
export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.25}
export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-0.45}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.75}

export CONTROL_LOSS_WEIGHT=${CONTROL_LOSS_WEIGHT:-0.0}
export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-9999}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.0}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.0}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.0}

export OVERWRITE=${OVERWRITE:-0}

exec bash configs/train_tfpp_tesla_town13_residual_policy_adapter_smoke_server.sh
