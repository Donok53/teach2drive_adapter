#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export COLLECT_CONFIG=${COLLECT_CONFIG:-configs/collect_tesla_pdm_lite_8town_3h.sh}
export CARLA_LOG=${CARLA_LOG:-"$HOME/teach2drive/logs/carla_pdm_lite_tesla_8town_3h_collect.log"}
export CARLA_READY_TIMEOUT_SEC=${CARLA_READY_TIMEOUT_SEC:-240}

exec bash configs/collect_autocarla_server.sh
