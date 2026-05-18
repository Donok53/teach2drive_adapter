#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Town13 expert imitation, v5.
# v4 showed good average open-loop metrics but failed closed-loop launch:
# direct-control logs kept predicting brake around 0.5 while the vehicle was
# stopped. This preset keeps expert control supervision, then adds explicit
# one-sided losses that penalize brake false-positives on go/launch samples and
# encourage minimum throttle when the expert is launching from low speed.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_vehicle_b_front_triplet_town13_target_3h"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_vehicle_b_front_triplet_town13_expert_v1"}
export OUT=${OUT:-"$HOME/teach2drive/runs/tfpp_vehicle_b_front_triplet_town13_expert_v5_launch_nobrake/train_front_triplet_shifted_town13_expert_v5_launch_nobrake"}

export CAMERAS=${CAMERAS:-left,front,right}
export EPOCHS=${EPOCHS:-80}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-15}
export BATCH_SIZE=${BATCH_SIZE:-128}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export OVERWRITE=${OVERWRITE:-0}

export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.25}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-1.0}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-2.0}

export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-1.0}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.05}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-1.0}
export SPEED_FLOOR_MASK=${SPEED_FLOOR_MASK:-target}
export SPEED_FLOOR_TARGET_THRESHOLD=${SPEED_FLOOR_TARGET_THRESHOLD:-2.0}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.15}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-2.0}
export LAUNCH_CURRENT_SPEED_THRESHOLD=${LAUNCH_CURRENT_SPEED_THRESHOLD:-0.8}
export LAUNCH_TARGET_SPEED_THRESHOLD=${LAUNCH_TARGET_SPEED_THRESHOLD:-2.0}
export LAUNCH_SAMPLE_WEIGHT=${LAUNCH_SAMPLE_WEIGHT:-4.0}
export LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=${LAUNCH_SPEED_FLOOR_LOSS_WEIGHT:-0.35}
export LAUNCH_SPEED_FLOOR_MPS=${LAUNCH_SPEED_FLOOR_MPS:-2.5}
export LAUNCH_STOP_NEGATIVE=${LAUNCH_STOP_NEGATIVE:-1}

export SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.10}
export SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.04}
export TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.05}
export TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.02}
export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.01}

# Control head: keep imitation, but make "do not brake when launching" a first
# class objective instead of hoping average SmoothL1 captures it.
export CONTROL_LOSS_WEIGHT=${CONTROL_LOSS_WEIGHT:-0.5}
export CONTROL_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT=${CONTROL_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT:-4.0}
export CONTROL_LAUNCH_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT=${CONTROL_LAUNCH_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT:-10.0}
export CONTROL_LAUNCH_THROTTLE_FLOOR_LOSS_WEIGHT=${CONTROL_LAUNCH_THROTTLE_FLOOR_LOSS_WEIGHT:-4.0}
export CONTROL_BRAKE_OFF_MAX=${CONTROL_BRAKE_OFF_MAX:-0.08}
export CONTROL_LAUNCH_THROTTLE_FLOOR=${CONTROL_LAUNCH_THROTTLE_FLOOR:-0.30}
export CONTROL_NO_BRAKE_TARGET_THRESHOLD=${CONTROL_NO_BRAKE_TARGET_THRESHOLD:-0.10}

# Keep stop classifiers out of the behavior path for this direct-control run.
export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-9999}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.0}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.0}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.0}

export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.0}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.0}

exec bash configs/train_tfpp_vehicle_b_expert_v1_server.sh
