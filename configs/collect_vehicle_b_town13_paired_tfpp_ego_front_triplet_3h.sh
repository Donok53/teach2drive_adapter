#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Target-ODD paired dataset for feature/fusion alignment:
#   - map/domain: Town13, matching the local leaderboard validation missions
#   - ego: vehicle.lincoln.mkz_2020
#   - paired rigs: tfpp_ego + front_triplet_shifted on the same ego/timestamps
#   - primary rig: front_triplet_shifted, matching the target vehicle-B sensor rig
#
# CARLA must already be running on HOST:PORT.

export CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_vehicle_b_town13_paired_tfpp_ego_front_triplet_3h"}

export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-2000}
export MAP=${MAP:-Town13}
export VEHICLE_FILTER=${VEHICLE_FILTER:-vehicle.lincoln.mkz_2020}

export PROFILES=${PROFILES:-tfpp_ego,front_triplet_shifted}
export PRIMARY_PROFILE=${PRIMARY_PROFILE:-front_triplet_shifted}
export COLLECTION_MODE=${COLLECTION_MODE:-paired}

export EPISODES=${EPISODES:-36}
export EPISODE_SEC=${EPISODE_SEC:-300}
export DURATION_HOURS=${DURATION_HOURS:-3.0}

export TRAFFIC_SCHEDULE=${TRAFFIC_SCHEDULE:-vehicle_b_mixed}
export TRAFFIC_SCHEDULE_SEED=${TRAFFIC_SCHEDULE_SEED:-13}
export TRAFFIC_VEHICLES=${TRAFFIC_VEHICLES:-60}
export GLOBAL_DISTANCE_TO_LEADING_VEHICLE=${GLOBAL_DISTANCE_TO_LEADING_VEHICLE:-2.5}
export GLOBAL_SPEED_DIFFERENCE=${GLOBAL_SPEED_DIFFERENCE:-0.0}
export IGNORE_LIGHTS_PERCENT=${IGNORE_LIGHTS_PERCENT:-0.0}

export FAIL_ON_INVALID_MOTION=${FAIL_ON_INVALID_MOTION:-1}
export MIN_MOVING_RATIO=${MIN_MOVING_RATIO:-0.20}
export MIN_PATH_LENGTH_M=${MIN_PATH_LENGTH_M:-100}
export MIN_MOVING_SPEED_MPS=${MIN_MOVING_SPEED_MPS:-1.0}

export LIDAR_FORMAT=${LIDAR_FORMAT:-npz}
export OVERWRITE=${OVERWRITE:-0}

exec bash configs/collect_transfuserpp_2h.sh
