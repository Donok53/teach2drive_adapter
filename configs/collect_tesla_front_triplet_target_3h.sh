#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Target-only expert collection.
# This deliberately records only the target rig:
#   - ego vehicle: vehicle.tesla.model3
#   - sensor rig: front_triplet_shifted only
#   - no tfpp_ego/canonical paired sensor collection
#
# The collector's "paired" mode is still used internally with a single profile
# because it gives the episode layout expected by the offline training tools.

export CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_front_triplet_target_3h"}

export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-2000}
export MAP=${MAP:-Town03}
export VEHICLE_FILTER=${VEHICLE_FILTER:-vehicle.tesla.model3}

export PROFILES=${PROFILES:-front_triplet_shifted}
export PRIMARY_PROFILE=${PRIMARY_PROFILE:-front_triplet_shifted}
export COLLECTION_MODE=${COLLECTION_MODE:-paired}

export EPISODES=${EPISODES:-36}
export EPISODE_SEC=${EPISODE_SEC:-300}
export DURATION_HOURS=${DURATION_HOURS:-3.0}
export TIMEOUT=${TIMEOUT:-180.0}
export SENSOR_TIMEOUT=${SENSOR_TIMEOUT:-10.0}

export TRAFFIC_SCHEDULE=${TRAFFIC_SCHEDULE:-vehicle_b_mixed}
export TRAFFIC_SCHEDULE_SEED=${TRAFFIC_SCHEDULE_SEED:-31}
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
