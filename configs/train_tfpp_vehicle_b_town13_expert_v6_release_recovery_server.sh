#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Town13 expert imitation, v6.
# v5 fixed most arbitrary stop-go and made hybrid longitudinal steering usable,
# but closed-loop failed after a legitimate stop: speed stayed near zero, stop
# probability stayed high, and the direct longitudinal head kept braking. This
# preset fine-tunes from v5 and explicitly upweights stop_state=release_go.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_vehicle_b_front_triplet_town13_target_3h"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_vehicle_b_front_triplet_expert_v1"}
export OUT=${OUT:-"$HOME/teach2drive/runs/tfpp_vehicle_b_front_triplet_town13_expert_v6_release_recovery/train_front_triplet_shifted_town13_expert_v6_release_recovery"}
export INIT_CHECKPOINT=${INIT_CHECKPOINT:-"$HOME/teach2drive/runs/tfpp_vehicle_b_front_triplet_town13_expert_v5_launch_nobrake/train_front_triplet_shifted_town13_expert_v5_launch_nobrake/best_model.pt"}

export CAMERAS=${CAMERAS:-left,front,right}
export EPOCHS=${EPOCHS:-60}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-15}
export BATCH_SIZE=${BATCH_SIZE:-128}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export OVERWRITE=${OVERWRITE:-0}
export LR=${LR:-5e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}

export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.15}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-1.0}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-2.0}

export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-0.9}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.03}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-1.2}
export SPEED_FLOOR_MASK=${SPEED_FLOOR_MASK:-target}
export SPEED_FLOOR_TARGET_THRESHOLD=${SPEED_FLOOR_TARGET_THRESHOLD:-1.5}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.08}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-1.5}

export LAUNCH_CURRENT_SPEED_THRESHOLD=${LAUNCH_CURRENT_SPEED_THRESHOLD:-1.2}
export LAUNCH_TARGET_SPEED_THRESHOLD=${LAUNCH_TARGET_SPEED_THRESHOLD:-1.0}
export LAUNCH_SAMPLE_WEIGHT=${LAUNCH_SAMPLE_WEIGHT:-4.0}
export LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=${LAUNCH_SPEED_FLOOR_LOSS_WEIGHT:-0.35}
export LAUNCH_SPEED_FLOOR_MPS=${LAUNCH_SPEED_FLOOR_MPS:-2.5}
export LAUNCH_STOP_NEGATIVE=${LAUNCH_STOP_NEGATIVE:-1}

export RELEASE_TARGET_SPEED_THRESHOLD=${RELEASE_TARGET_SPEED_THRESHOLD:-1.0}
export RELEASE_SAMPLE_WEIGHT=${RELEASE_SAMPLE_WEIGHT:-8.0}
export RELEASE_SPEED_FLOOR_LOSS_WEIGHT=${RELEASE_SPEED_FLOOR_LOSS_WEIGHT:-0.75}
export RELEASE_SPEED_FLOOR_MPS=${RELEASE_SPEED_FLOOR_MPS:-2.5}
export RELEASE_STOP_NEGATIVE=${RELEASE_STOP_NEGATIVE:-1}

export SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.08}
export SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.03}
export TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.04}
export TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.02}
export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.005}

# Hybrid inference uses PID steering, so the control head should mostly learn
# longitudinal release behavior. Keep a modest overall imitation term, then add
# stronger no-brake/throttle terms specifically for launch and release_go.
export CONTROL_LOSS_WEIGHT=${CONTROL_LOSS_WEIGHT:-0.35}
export CONTROL_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT=${CONTROL_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT:-1.5}
export CONTROL_LAUNCH_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT=${CONTROL_LAUNCH_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT:-8.0}
export CONTROL_LAUNCH_THROTTLE_FLOOR_LOSS_WEIGHT=${CONTROL_LAUNCH_THROTTLE_FLOOR_LOSS_WEIGHT:-4.0}
export CONTROL_RELEASE_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT=${CONTROL_RELEASE_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT:-14.0}
export CONTROL_RELEASE_THROTTLE_FLOOR_LOSS_WEIGHT=${CONTROL_RELEASE_THROTTLE_FLOOR_LOSS_WEIGHT:-8.0}
export CONTROL_BRAKE_OFF_MAX=${CONTROL_BRAKE_OFF_MAX:-0.06}
export CONTROL_LAUNCH_THROTTLE_FLOOR=${CONTROL_LAUNCH_THROTTLE_FLOOR:-0.30}
export CONTROL_RELEASE_THROTTLE_FLOOR=${CONTROL_RELEASE_THROTTLE_FLOOR:-0.35}
export CONTROL_NO_BRAKE_TARGET_THRESHOLD=${CONTROL_NO_BRAKE_TARGET_THRESHOLD:-0.12}

# Keep light stop supervision so shared features still separate stop/wait/release,
# but do not let it dominate the behavior heads.
export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-1}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.04}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.05}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.02}
export STOP_POSITIVE_LOSS_SCALE=${STOP_POSITIVE_LOSS_SCALE:-1.0}
export STOP_NEGATIVE_LOSS_SCALE=${STOP_NEGATIVE_LOSS_SCALE:-2.0}

export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.0}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.0}

exec bash configs/train_tfpp_vehicle_b_expert_v1_server.sh
