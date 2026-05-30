#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SESSION="${SESSION:-pf}"
PORT="${PORT:-2007}"
TM_PORT="${TM_PORT:-8007}"
CARLA_GRAPHICS_ADAPTER="${CARLA_GRAPHICS_ADAPTER:-7}"
CARLA_QUALITY_LEVEL="${CARLA_QUALITY_LEVEL:-Low}"
CARLA_EXTRA_ARGS="${CARLA_EXTRA_ARGS:--stdout -FullStdOutLogOutput}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/jovyan/dataset/byeongjae/datasets/pdm_lite_tesla_front_triplet_8town_3h}"
WORK_ROOT="${WORK_ROOT:-/home/jovyan/dataset/byeongjae/runs/pdm_lite_tesla_front_triplet_8town_3h_collect}"
DATASET_NAME="${DATASET_NAME:-pdm_lite_tesla_front_triplet_8town_3h}"
RUN_LOG="${RUN_LOG:-/home/jovyan/teach2drive/logs/collect_pdm_lite_tesla_front_triplet_supervised_port${PORT}.log}"
COLLECT_CONFIG="${COLLECT_CONFIG:-/tmp/cfg.sh}"

mkdir -p "$(dirname "$RUN_LOG")"

tmux kill-session -t "$SESSION" 2>/dev/null || true

for port in "$PORT" 2006 2007; do
  pids=$(pgrep -f "leaderboard_evaluator_local.py.*--port ${port}|carla-rpc-port=${port}" || true)
  if [[ -n "$pids" ]]; then
    echo "stopping stale collection processes on port $port: ${pids//$'\n'/ }"
    kill -TERM $pids 2>/dev/null || true
    sleep 2
    pids=$(pgrep -f "leaderboard_evaluator_local.py.*--port ${port}|carla-rpc-port=${port}" || true)
    [[ -n "$pids" ]] && kill -KILL $pids 2>/dev/null || true
  fi
done

tmux new-session -d -s "$SESSION" bash -lc \
  "cd '$PWD' && \
   OUTPUT_ROOT='$OUTPUT_ROOT' \
   WORK_ROOT='$WORK_ROOT' \
   DATASET_NAME='$DATASET_NAME' \
   RUN_LOG='$RUN_LOG' \
   COLLECT_CONFIG='$COLLECT_CONFIG' \
   ROUTE_START='0' \
   ROUTE_END='385' \
   ROUTE_SKIP_CSV='' \
   MAX_ROUTE_ATTEMPTS='1000' \
   PORT='$PORT' \
   TM_PORT='$TM_PORT' \
   CARLA_GRAPHICS_ADAPTER='$CARLA_GRAPHICS_ADAPTER' \
   CARLA_QUALITY_LEVEL='$CARLA_QUALITY_LEVEL' \
   CARLA_EXTRA_ARGS='$CARLA_EXTRA_ARGS' \
   ROUTE_TIMEOUT_SEC='720' \
   ROUTE_TIMEOUT_KILL_AFTER_SEC='60' \
   bash configs/collect_tesla_pdm_lite_8town_3h_supervised_autocarla_server.sh"

echo "started collection session=$SESSION port=$PORT tm_port=$TM_PORT gpu=$CARLA_GRAPHICS_ADAPTER quality=$CARLA_QUALITY_LEVEL args=$CARLA_EXTRA_ARGS log=$RUN_LOG"
tmux ls
