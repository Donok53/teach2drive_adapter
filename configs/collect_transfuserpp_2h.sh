#!/usr/bin/env bash
set -euo pipefail

# CARLA server must already be running on HOST:PORT.
# The default command collects 2.0 simulated hours on one ego trajectory.
# Each saved frame contains both:
#   1. tfpp_ego: official TransFuser++ single-front-camera rig
#   2. front_triplet_shifted: plausible alternate-ego front triplet + roof LiDAR rig

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-2000}
CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/t2d_transfuserpp_2h"}
MAP=${MAP:-Town10HD_Opt}
DURATION_HOURS=${DURATION_HOURS:-2.0}
EPISODE_SEC=${EPISODE_SEC:-300}
TRAFFIC_VEHICLES=${TRAFFIC_VEHICLES:-60}
GLOBAL_DISTANCE_TO_LEADING_VEHICLE=${GLOBAL_DISTANCE_TO_LEADING_VEHICLE:-2.5}
GLOBAL_SPEED_DIFFERENCE=${GLOBAL_SPEED_DIFFERENCE:-0.0}
IGNORE_LIGHTS_PERCENT=${IGNORE_LIGHTS_PERCENT:-0.0}
MIN_MOVING_SPEED_MPS=${MIN_MOVING_SPEED_MPS:-1.0}
MIN_MOVING_RATIO=${MIN_MOVING_RATIO:-0.20}
MIN_PATH_LENGTH_M=${MIN_PATH_LENGTH_M:-100.0}
FAIL_ON_INVALID_MOTION=${FAIL_ON_INVALID_MOTION:-0}
PROFILES=${PROFILES:-tfpp_ego,front_triplet_shifted}
COLLECTION_MODE=${COLLECTION_MODE:-paired}
PRIMARY_PROFILE=${PRIMARY_PROFILE:-tfpp_ego}
COMMAND_POLICY=${COMMAND_POLICY:-heuristic}
EPISODES=${EPISODES:-0}
START_EPISODE_INDEX=${START_EPISODE_INDEX:-0}
LIDAR_FORMAT=${LIDAR_FORMAT:-npz}
OVERWRITE=${OVERWRITE:-0}

EXTRA_ARGS=()
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ "$EPISODES" != "0" ]]; then
  EXTRA_ARGS+=(--episodes "$EPISODES")
fi
if [[ "$START_EPISODE_INDEX" != "0" ]]; then
  EXTRA_ARGS+=(--start-episode-index "$START_EPISODE_INDEX")
fi
if [[ "$FAIL_ON_INVALID_MOTION" == "1" || "$FAIL_ON_INVALID_MOTION" == "true" || "$FAIL_ON_INVALID_MOTION" == "TRUE" ]]; then
  EXTRA_ARGS+=(--fail-on-invalid-motion)
fi

python -m teach2drive_adapter.collect_transfuserpp_dataset \
  --host "$HOST" \
  --port "$PORT" \
  --carla-root "$CARLA_ROOT" \
  --map "$MAP" \
  --output-root "$OUTPUT_ROOT" \
  --profiles "$PROFILES" \
  --collection-mode "$COLLECTION_MODE" \
  --primary-profile "$PRIMARY_PROFILE" \
  --command-policy "$COMMAND_POLICY" \
  --duration-hours "$DURATION_HOURS" \
  --episode-sec "$EPISODE_SEC" \
  --hz 20 \
  --save-every-n 5 \
  --traffic-vehicles "$TRAFFIC_VEHICLES" \
  --global-distance-to-leading-vehicle "$GLOBAL_DISTANCE_TO_LEADING_VEHICLE" \
  --global-speed-difference "$GLOBAL_SPEED_DIFFERENCE" \
  --ignore-lights-percent "$IGNORE_LIGHTS_PERCENT" \
  --min-moving-speed-mps "$MIN_MOVING_SPEED_MPS" \
  --min-moving-ratio "$MIN_MOVING_RATIO" \
  --min-path-length-m "$MIN_PATH_LENGTH_M" \
  --lidar-format "$LIDAR_FORMAT" \
  "${EXTRA_ARGS[@]}"
