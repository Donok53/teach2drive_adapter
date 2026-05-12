#!/usr/bin/env bash
set -euo pipefail

# CARLA server must already be running on HOST:PORT.
# The default command collects 2.5 simulated hours total, split evenly between:
#   1. tfpp_ego: official TransFuser++ single-front-camera rig
#   2. front_triplet_shifted: three front-facing cameras with shifted poses

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-2000}
CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/t2d_transfuserpp_2h"}
MAP=${MAP:-Town10HD_Opt}
DURATION_HOURS=${DURATION_HOURS:-2.5}
EPISODE_SEC=${EPISODE_SEC:-300}
TRAFFIC_VEHICLES=${TRAFFIC_VEHICLES:-60}
PROFILES=${PROFILES:-tfpp_ego,front_triplet_shifted}
LIDAR_FORMAT=${LIDAR_FORMAT:-npz}
OVERWRITE=${OVERWRITE:-0}

EXTRA_ARGS=()
if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

python -m teach2drive_adapter.collect_transfuserpp_dataset \
  --host "$HOST" \
  --port "$PORT" \
  --carla-root "$CARLA_ROOT" \
  --map "$MAP" \
  --output-root "$OUTPUT_ROOT" \
  --profiles "$PROFILES" \
  --duration-hours "$DURATION_HOURS" \
  --episode-sec "$EPISODE_SEC" \
  --hz 20 \
  --save-every-n 5 \
  --traffic-vehicles "$TRAFFIC_VEHICLES" \
  --lidar-format "$LIDAR_FORMAT" \
  "${EXTRA_ARGS[@]}"
