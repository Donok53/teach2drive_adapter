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
PROFILES=${PROFILES:-tfpp_ego,front_triplet_shifted}
COLLECTION_MODE=${COLLECTION_MODE:-paired}
PRIMARY_PROFILE=${PRIMARY_PROFILE:-tfpp_ego}
COMMAND_POLICY=${COMMAND_POLICY:-heuristic}
EPISODES=${EPISODES:-0}
LIDAR_FORMAT=${LIDAR_FORMAT:-npz}
OVERWRITE=${OVERWRITE:-0}

EXTRA_ARGS=()
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ "$EPISODES" != "0" ]]; then
  EXTRA_ARGS+=(--episodes "$EPISODES")
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
  --lidar-format "$LIDAR_FORMAT" \
  "${EXTRA_ARGS[@]}"
