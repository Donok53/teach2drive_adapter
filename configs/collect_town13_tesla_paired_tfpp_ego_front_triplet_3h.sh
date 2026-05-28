#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Town13-only domain shift collection.
# Keep the original best-dataset vehicle and sensor setup:
#   - ego vehicle: vehicle.tesla.model3
#   - paired rigs: tfpp_ego + front_triplet_shifted
#   - primary profile: tfpp_ego
# Only the map / ODD is changed to Town13.
#
# CARLA must already be running on HOST:PORT.

export CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_town13_paired_tfpp_ego_front_triplet_3h"}

export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-2000}
export MAP=${MAP:-Town13}
export VEHICLE_FILTER=${VEHICLE_FILTER:-vehicle.tesla.model3}

export PROFILES=${PROFILES:-tfpp_ego,front_triplet_shifted}
export PRIMARY_PROFILE=${PRIMARY_PROFILE:-tfpp_ego}
export COLLECTION_MODE=${COLLECTION_MODE:-paired}

export EPISODES=${EPISODES:-36}
export EPISODE_SEC=${EPISODE_SEC:-300}
export DURATION_HOURS=${DURATION_HOURS:-3.0}
export TIMEOUT=${TIMEOUT:-180.0}

export TRAFFIC_SCHEDULE=${TRAFFIC_SCHEDULE:-fixed}
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
