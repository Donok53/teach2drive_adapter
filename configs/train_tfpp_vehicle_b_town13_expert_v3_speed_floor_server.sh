#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Town13 expert imitation, v3.
# v2 removed the startup deadlock, but still produced stop-go oscillation.
# The main fix here is to apply speed-floor pressure only when the expert
# future-speed label says "go"; true stop frames are no longer pulled upward by
# the pretrained base prior's high speed.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_vehicle_b_front_triplet_town13_target_3h"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_vehicle_b_front_triplet_town13_expert_v1"}
export OUT=${OUT:-"$HOME/teach2drive/runs/tfpp_vehicle_b_front_triplet_town13_expert_v3_speed_floor/train_front_triplet_shifted_town13_expert_v3_speed_floor"}

export CAMERAS=${CAMERAS:-left,front,right}
export EPOCHS=${EPOCHS:-80}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-15}
export BATCH_SIZE=${BATCH_SIZE:-128}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export OVERWRITE=${OVERWRITE:-0}

export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.35}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-1.0}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-2.0}

export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-1.3}
export SPEED_FLOOR_MASK=${SPEED_FLOOR_MASK:-target}
export SPEED_FLOOR_TARGET_THRESHOLD=${SPEED_FLOOR_TARGET_THRESHOLD:-2.0}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.35}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-2.5}
export LAUNCH_CURRENT_SPEED_THRESHOLD=${LAUNCH_CURRENT_SPEED_THRESHOLD:-0.8}
export LAUNCH_TARGET_SPEED_THRESHOLD=${LAUNCH_TARGET_SPEED_THRESHOLD:-2.0}
export LAUNCH_SAMPLE_WEIGHT=${LAUNCH_SAMPLE_WEIGHT:-3.0}
export LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=${LAUNCH_SPEED_FLOOR_LOSS_WEIGHT:-0.4}
export LAUNCH_SPEED_FLOOR_MPS=${LAUNCH_SPEED_FLOOR_MPS:-3.0}
export LAUNCH_STOP_NEGATIVE=${LAUNCH_STOP_NEGATIVE:-1}

# Do not let the auxiliary stop classifiers shape the shared adapter features
# until the speed policy is stable. We evaluate this preset with stop gate off.
export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-9999}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.0}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.0}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.0}

export SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.12}
export SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.05}
export TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.05}
export TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.02}
export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.03}
export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.0}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.0}

exec bash configs/train_tfpp_vehicle_b_expert_v1_server.sh
