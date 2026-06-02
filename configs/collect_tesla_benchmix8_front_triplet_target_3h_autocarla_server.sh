#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 3h target-only expert collection over the benchmark-aligned 8-town mix.
# This is a thin preset wrapper around the generic multitown collector.

export PY=${PY:-"$HOME/.venv/carla37/bin/python"}
export CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-2007}

export TOWN_PRESET=${TOWN_PRESET:-benchmix8}
export TOWNS_CSV=${TOWNS_CSV:-Town01,Town02,Town03,Town04,Town05,Town06,Town10HD,Town12}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_benchmix8_front_triplet_target_3h"}
export TOTAL_EPISODES=${TOTAL_EPISODES:-36}
export EPISODE_SEC=${EPISODE_SEC:-300}

export VEHICLE_FILTER=${VEHICLE_FILTER:-vehicle.tesla.model3}
export PROFILES=front_triplet_shifted
export PRIMARY_PROFILE=front_triplet_shifted
export COLLECTION_MODE=paired

export CARLA_GRAPHICS_ADAPTER=${CARLA_GRAPHICS_ADAPTER:-6}
export CARLA_EXTRA_ARGS=${CARLA_EXTRA_ARGS:-"-nothreadtimeout -stdout -FullStdOutLogOutput"}
export CARLA_LOG=${CARLA_LOG:-"$HOME/teach2drive/logs/carla_benchmix8_tesla_front_triplet_collect.log"}
export CARLA_PRELOAD_MAP=${CARLA_PRELOAD_MAP:-1}
export COLLECT_SKIP_LOAD_WORLD=${COLLECT_SKIP_LOAD_WORLD:-1}
export FORCE_RESTART_CARLA=${FORCE_RESTART_CARLA:-1}
export CARLA_MAP_LOAD_TIMEOUT_SEC=${CARLA_MAP_LOAD_TIMEOUT_SEC:-240}
export CARLA_MAP_RETRIES=${CARLA_MAP_RETRIES:-5}
export CARLA_READY_TIMEOUT_SEC=${CARLA_READY_TIMEOUT_SEC:-180}
export CARLA_POST_MAP_SETTLE_SEC=${CARLA_POST_MAP_SETTLE_SEC:-12}

export TRAFFIC_SCHEDULE=${TRAFFIC_SCHEDULE:-vehicle_b_mixed}
export TRAFFIC_VEHICLES=${TRAFFIC_VEHICLES:-60}
export FAIL_ON_INVALID_MOTION=${FAIL_ON_INVALID_MOTION:-1}
export CONTINUE_ON_TOWN_FAILURE=${CONTINUE_ON_TOWN_FAILURE:-1}

exec bash configs/collect_tesla_multitown_front_triplet_target_3h_autocarla_server.sh
