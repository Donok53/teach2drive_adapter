#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Town13 target-domain expert imitation with an explicit launch/recovery prior.
# The v1 expert model matched open-loop speed, but closed-loop rollout often fell
# into a zero-speed attractor. This preset upweights frames where the car is
# currently slow but the expert future speed says it should move.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_vehicle_b_front_triplet_town13_target_3h"}

# Reuse the v1 exported views/index/cache by default so retraining starts quickly.
export WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_vehicle_b_front_triplet_town13_expert_v1"}
export OUT=${OUT:-"$HOME/teach2drive/runs/tfpp_vehicle_b_front_triplet_town13_expert_v2_launch/train_front_triplet_shifted_town13_expert_v2_launch"}

export CAMERAS=${CAMERAS:-left,front,right}
export EPOCHS=${EPOCHS:-80}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-15}
export BATCH_SIZE=${BATCH_SIZE:-128}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export OVERWRITE=${OVERWRITE:-0}

# Keep the expert policy dominant, but stop treating "currently stopped" as a
# reason to predict zero when future expert speed indicates launch.
export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.8}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-0.35}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-3.0}

export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-1.2}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.05}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-1.5}
export LAUNCH_CURRENT_SPEED_THRESHOLD=${LAUNCH_CURRENT_SPEED_THRESHOLD:-0.8}
export LAUNCH_TARGET_SPEED_THRESHOLD=${LAUNCH_TARGET_SPEED_THRESHOLD:-2.0}
export LAUNCH_SAMPLE_WEIGHT=${LAUNCH_SAMPLE_WEIGHT:-5.0}
export LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=${LAUNCH_SPEED_FLOOR_LOSS_WEIGHT:-1.0}
export LAUNCH_SPEED_FLOOR_MPS=${LAUNCH_SPEED_FLOOR_MPS:-3.0}
export LAUNCH_STOP_NEGATIVE=${LAUNCH_STOP_NEGATIVE:-1}

# Delay and weaken stop losses. Evaluate v2 first with TFPP_ADAPTER_USE_STOP=0;
# only re-enable the stop gate after the speed head can reliably launch.
export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-10}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.02}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.05}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.02}
export STOP_POSITIVE_LOSS_SCALE=${STOP_POSITIVE_LOSS_SCALE:-1.0}
export STOP_NEGATIVE_LOSS_SCALE=${STOP_NEGATIVE_LOSS_SCALE:-1.5}

export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.03}
export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.0}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.0}

exec bash configs/train_tfpp_vehicle_b_expert_v1_server.sh
