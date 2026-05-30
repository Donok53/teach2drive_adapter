#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Robust PDM-Lite Tesla collection runner.
#
# The leaderboard evaluator can leave the whole collection stopped when CARLA
# segfaults while loading a route. This supervisor runs one route per evaluator
# process, restarts CARLA for every route, retries failures, and records per-route
# status so the collection can resume without redoing completed routes.

export PY=${PY:-"$HOME/.venv/carla37/bin/python"}
export CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/datasets/pdm_lite_tesla_8town_3h"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/pdm_lite_tesla_8town_3h_collect"}
export DATASET_NAME=${DATASET_NAME:-pdm_lite_tesla_8town_3h}
export EGO_VEHICLE_MODEL=${EGO_VEHICLE_MODEL:-vehicle.tesla.model3}
export ROUTE_START=${ROUTE_START:-4}
export ROUTE_END=${ROUTE_END:-385}
export ROUTE_SKIP_CSV=${ROUTE_SKIP_CSV:-2}
export MAX_ROUTE_ATTEMPTS=${MAX_ROUTE_ATTEMPTS:-2}
export PORT=${PORT:-2006}
export TM_PORT=${TM_PORT:-8006}
export CARLA_GRAPHICS_ADAPTER=${CARLA_GRAPHICS_ADAPTER:-6}
export ROUTE_TIMEOUT_SEC=${ROUTE_TIMEOUT_SEC:-900}
export ROUTE_TIMEOUT_KILL_AFTER_SEC=${ROUTE_TIMEOUT_KILL_AFTER_SEC:-60}
export RUN_LOG=${RUN_LOG:-"$HOME/teach2drive/logs/collect_pdm_lite_tesla_supervised_port${PORT}.log"}
export SUPERVISOR_STATE_DIR=${SUPERVISOR_STATE_DIR:-"$OUTPUT_ROOT/results/supervised_routes"}
export COLLECT_CONFIG=${COLLECT_CONFIG:-configs/collect_tesla_pdm_lite_8town_3h.sh}
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$RUN_LOG")" "$SUPERVISOR_STATE_DIR" "$OUTPUT_ROOT/results"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$RUN_LOG"
}

route_is_skipped() {
  local route="$1"
  local item
  IFS=',' read -ra items <<< "$ROUTE_SKIP_CSV"
  for item in "${items[@]}"; do
    [[ -n "$item" && "$item" == "$route" ]] && return 0
  done
  return 1
}

kill_carla_on_port() {
  local port="$1"
  local pids
  pids=$(pgrep -f "carla-rpc-port=${port}" || true)
  if [[ -n "$pids" ]]; then
    log "stopping stale CARLA on port ${port}: ${pids//$'\n'/ }"
    kill -TERM $pids 2>/dev/null || true
    sleep 5
    pids=$(pgrep -f "carla-rpc-port=${port}" || true)
    if [[ -n "$pids" ]]; then
      log "force stopping stale CARLA on port ${port}: ${pids//$'\n'/ }"
      kill -KILL $pids 2>/dev/null || true
      sleep 2
    fi
  fi
}

checkpoint_has_record() {
  local checkpoint="$1"
  [[ -s "$checkpoint" ]] || return 1
  python3 - "$checkpoint" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("_checkpoint", {}).get("records", [])
except Exception:
    sys.exit(1)
sys.exit(0 if records else 1)
PY
}

log "supervised PDM-Lite Tesla collection start"
log "routes=${ROUTE_START}-${ROUTE_END} skip=${ROUTE_SKIP_CSV} port=${PORT} tm_port=${TM_PORT} gpu=${CARLA_GRAPHICS_ADAPTER}"
log "route_timeout=${ROUTE_TIMEOUT_SEC}s kill_after=${ROUTE_TIMEOUT_KILL_AFTER_SEC}s dataset=${DATASET_NAME}"
log "output=${OUTPUT_ROOT}"

for route in $(seq "$ROUTE_START" "$ROUTE_END"); do
  done_file="$SUPERVISOR_STATE_DIR/route_${route}.done"
  fail_file="$SUPERVISOR_STATE_DIR/route_${route}.failed"
  checkpoint="$OUTPUT_ROOT/results/${DATASET_NAME}_route_${route}_result.json"
  debug_checkpoint="$OUTPUT_ROOT/results/${DATASET_NAME}_route_${route}_live.txt"
  carla_log="$HOME/teach2drive/logs/carla_pdm_lite_tesla_route_${route}_port${PORT}.log"

  if route_is_skipped "$route"; then
    log "route ${route}: skipped by ROUTE_SKIP_CSV"
    continue
  fi

  if [[ -f "$done_file" ]] || checkpoint_has_record "$checkpoint"; then
    touch "$done_file"
    log "route ${route}: already completed"
    continue
  fi

  rm -f "$fail_file"
  success=0
  for attempt in $(seq 1 "$MAX_ROUTE_ATTEMPTS"); do
    log "route ${route}: attempt ${attempt}/${MAX_ROUTE_ATTEMPTS}"
    kill_carla_on_port "$PORT"

    set +e
    timeout --kill-after="${ROUTE_TIMEOUT_KILL_AFTER_SEC}s" "${ROUTE_TIMEOUT_SEC}s" env \
      PY="$PY" \
      HOST=127.0.0.1 \
      PORT="$PORT" \
      TM_PORT="$TM_PORT" \
      CARLA_ROOT="$CARLA_ROOT" \
      GARAGE_ROOT="$GARAGE_ROOT" \
      OUTPUT_ROOT="$OUTPUT_ROOT" \
      WORK_ROOT="$WORK_ROOT" \
      EGO_VEHICLE_MODEL="$EGO_VEHICLE_MODEL" \
      ROUTES_SUBSET="${route}-${route}" \
      RESUME=0 \
      CHECKPOINT_ENDPOINT="$checkpoint" \
      DEBUG_CHECKPOINT="$debug_checkpoint" \
      CARLA_GRAPHICS_ADAPTER="$CARLA_GRAPHICS_ADAPTER" \
      CARLA_LOG="$carla_log" \
      COLLECT_CONFIG="$COLLECT_CONFIG" \
      bash configs/collect_autocarla_server.sh >> "$RUN_LOG" 2>&1
    rc=$?
    set -e

    kill_carla_on_port "$PORT"

    if [[ "$rc" == "0" ]] && checkpoint_has_record "$checkpoint"; then
      touch "$done_file"
      log "route ${route}: completed"
      success=1
      break
    fi

    log "route ${route}: attempt ${attempt} failed rc=${rc}"
    sleep 5
  done

  if [[ "$success" != "1" ]]; then
    printf 'route=%s failed_at=%s\n' "$route" "$(date '+%Y-%m-%d %H:%M:%S')" > "$fail_file"
    log "route ${route}: failed after ${MAX_ROUTE_ATTEMPTS} attempts; continuing"
  fi
done

kill_carla_on_port "$PORT"
done_count=$(find "$SUPERVISOR_STATE_DIR" -maxdepth 1 -name 'route_*.done' | wc -l | tr -d ' ')
fail_count=$(find "$SUPERVISOR_STATE_DIR" -maxdepth 1 -name 'route_*.failed' | wc -l | tr -d ' ')
log "supervised collection finished done=${done_count} failed=${fail_count}"
