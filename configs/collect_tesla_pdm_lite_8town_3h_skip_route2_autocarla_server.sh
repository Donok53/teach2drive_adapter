#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export RUN_LOG=${RUN_LOG:-"$HOME/teach2drive/logs/collect_pdm_lite_tesla_skip_route2_port2002.log"}
if [[ "${_T2D_SKIP_ROUTE2_LOGGED:-0}" != "1" ]]; then
  mkdir -p "$(dirname "$RUN_LOG")"
  export _T2D_SKIP_ROUTE2_LOGGED=1
  exec bash "$0" "$@" > "$RUN_LOG" 2>&1
fi

export PY=${PY:-"$HOME/.venv/carla37/bin/python"}
export CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/datasets/pdm_lite_tesla_8town_3h"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/pdm_lite_tesla_8town_3h_collect"}
export EGO_VEHICLE_MODEL=${EGO_VEHICLE_MODEL:-vehicle.tesla.model3}

# Route 2 repeatedly crashes CARLA during world/scenario loading on the current
# server. Keep route0/route1 data and continue from route3 in a separate
# checkpoint so the full collection can keep making progress.
export PORT=${PORT:-2002}
export TM_PORT=${TM_PORT:-8002}
export ROUTES_SUBSET=${ROUTES_SUBSET:-3-385}
export RESUME=${RESUME:-0}
export CHECKPOINT_ENDPOINT=${CHECKPOINT_ENDPOINT:-"$OUTPUT_ROOT/results/pdm_lite_tesla_8town_3h_skip_route2_result.json"}
export DEBUG_CHECKPOINT=${DEBUG_CHECKPOINT:-"$OUTPUT_ROOT/results/pdm_lite_tesla_8town_3h_skip_route2_live.txt"}
export CARLA_GRAPHICS_ADAPTER=${CARLA_GRAPHICS_ADAPTER:-7}
export CARLA_LOG=${CARLA_LOG:-"$HOME/teach2drive/logs/carla_pdm_lite_tesla_skip_route2_port2002.log"}
export COLLECT_CONFIG=${COLLECT_CONFIG:-configs/collect_tesla_pdm_lite_8town_3h.sh}

exec bash configs/collect_autocarla_server.sh
