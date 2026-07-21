#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# One-route-at-a-time full collection supervisor for the local Docker runtime.

export PY=${PY:-python3}
export CARLA_ROOT=${CARLA_ROOT:-"$HOME/carla-simulator"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/code/carla_garage"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/datasets/pdm_lite_tesla_front_triplet_shifted_full"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/pdm_lite_tesla_front_triplet_shifted_full_collect"}
export DATASET_NAME=${DATASET_NAME:-pdm_lite_tesla_front_triplet_shifted_full}
export EGO_VEHICLE_MODEL=${EGO_VEHICLE_MODEL:-vehicle.tesla.model3}
export TFPP_SENSOR_RIG=${TFPP_SENSOR_RIG:-front_triplet_shifted}
export TFPP_SENSOR_CAMERA=${TFPP_SENSOR_CAMERA:-front}
export TFPP_SENSOR_LIDAR=${TFPP_SENSOR_LIDAR:-top}
export ROUTE_START=${ROUTE_START:-0}
export ROUTE_END=${ROUTE_END:-}
export ROUTE_SKIP_CSV=${ROUTE_SKIP_CSV:-}
export MAX_ROUTE_ATTEMPTS=${MAX_ROUTE_ATTEMPTS:-10}
export PORT=${PORT:-2000}
export TM_PORT=${TM_PORT:-8000}
export CARLA_GRAPHICS_ADAPTER=${CARLA_GRAPHICS_ADAPTER:-1}
export ROUTE_TIMEOUT_SEC=${ROUTE_TIMEOUT_SEC:-900}
export ROUTE_TIMEOUT_KILL_AFTER_SEC=${ROUTE_TIMEOUT_KILL_AFTER_SEC:-60}
export RUN_LOG=${RUN_LOG:-"$HOME/teach2drive/logs/collect_${DATASET_NAME}_local_port${PORT}.log"}
export SUPERVISOR_STATE_DIR=${SUPERVISOR_STATE_DIR:-"$OUTPUT_ROOT/results/supervised_routes"}
export COLLECT_CONFIG=${COLLECT_CONFIG:-configs/collect_tesla_pdm_lite_front_triplet_full.sh}
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$RUN_LOG")" "$SUPERVISOR_STATE_DIR" "$OUTPUT_ROOT/results" "$WORK_ROOT/routes"

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

checkpoint_is_successful() {
  local checkpoint="$1"
  [[ -s "$checkpoint" ]] || return 1
  "$PY" - "$checkpoint" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("_checkpoint", {}).get("records", [])
except Exception:
    sys.exit(1)
if not records:
    sys.exit(1)
for record in records:
    status = str(record.get("status", ""))
    if status.startswith("Failed"):
        sys.exit(1)
sys.exit(0)
PY
}

ensure_route_subset() {
  REFRESH_ROUTE_SUBSET=${REFRESH_ROUTE_SUBSET:-0} "$PY" - <<'PY'
import os
import subprocess
from pathlib import Path

adapter = Path.cwd()
work = Path(os.environ["WORK_ROOT"])
xml = Path(os.environ.get("ROUTE_SUBSET_XML", work / "routes" / f"{os.environ['DATASET_NAME']}_routes.xml"))
meta = Path(os.environ.get("ROUTE_SUBSET_META", work / "routes" / f"{os.environ['DATASET_NAME']}_routes.json"))
if os.environ.get("REFRESH_ROUTE_SUBSET", "0") == "1" or not xml.exists() or xml.stat().st_size == 0:
    xml.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        os.environ.get("PY", "python3"),
        str(adapter / "scripts" / "build_pdm_lite_route_subset.py"),
        "--route-root", os.environ.get("ROUTE_ROOT", str(Path(os.environ["GARAGE_ROOT"]) / "data")),
        "--output", str(xml),
        "--meta-output", str(meta),
        "--towns-csv", os.environ.get("TOWNS_CSV", "Town01,Town02,Town03,Town04,Town05,Town10HD,Town12,Town13"),
        "--total-routes", os.environ.get("TOTAL_ROUTES", "999999"),
        "--seed", os.environ.get("ROUTE_SEED", "42"),
    ]
    subprocess.check_call(cmd)
PY
}

selected_last_route() {
  "$PY" - <<'PY'
import json
import os
from pathlib import Path
work = Path(os.environ["WORK_ROOT"])
meta = Path(os.environ.get("ROUTE_SUBSET_META", work / "routes" / f"{os.environ['DATASET_NAME']}_routes.json"))
with open(meta, "r", encoding="utf-8") as f:
    total = int(json.load(f)["selected_total_routes"])
print(total - 1)
PY
}

ensure_route_subset
if [[ -z "$ROUTE_END" ]]; then
  ROUTE_END=$(selected_last_route)
  export ROUTE_END
fi

log "supervised full PDM-Lite Tesla front_triplet_shifted collection start"
log "routes=${ROUTE_START}-${ROUTE_END} skip=${ROUTE_SKIP_CSV:-<none>}"
log "port=${PORT} tm_port=${TM_PORT} gpu=${CARLA_GRAPHICS_ADAPTER}"
log "route_timeout=${ROUTE_TIMEOUT_SEC}s kill_after=${ROUTE_TIMEOUT_KILL_AFTER_SEC}s dataset=${DATASET_NAME}"
log "output=${OUTPUT_ROOT}"

for route in $(seq "$ROUTE_START" "$ROUTE_END"); do
  done_file="$SUPERVISOR_STATE_DIR/route_${route}.done"
  fail_file="$SUPERVISOR_STATE_DIR/route_${route}.failed"
  checkpoint="$OUTPUT_ROOT/results/${DATASET_NAME}_route_${route}_result.json"
  debug_checkpoint="$OUTPUT_ROOT/results/${DATASET_NAME}_route_${route}_live.txt"
  carla_log="$HOME/teach2drive/logs/carla_${DATASET_NAME}_route_${route}_port${PORT}.log"

  if route_is_skipped "$route"; then
    log "route ${route}: skipped by ROUTE_SKIP_CSV"
    continue
  fi

  if checkpoint_is_successful "$checkpoint"; then
    touch "$done_file"
    log "route ${route}: already completed"
    continue
  fi

  rm -f "$done_file" "$fail_file"
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
      DATASET_NAME="$DATASET_NAME" \
      EGO_VEHICLE_MODEL="$EGO_VEHICLE_MODEL" \
      TFPP_SENSOR_RIG="$TFPP_SENSOR_RIG" \
      TFPP_SENSOR_CAMERA="$TFPP_SENSOR_CAMERA" \
      TFPP_SENSOR_LIDAR="$TFPP_SENSOR_LIDAR" \
      ROUTES_SUBSET="${route}-${route}" \
      RESUME=0 \
      CHECKPOINT_ENDPOINT="$checkpoint" \
      DEBUG_CHECKPOINT="$debug_checkpoint" \
      CARLA_GRAPHICS_ADAPTER="$CARLA_GRAPHICS_ADAPTER" \
      CARLA_RUN_AS_UID="${CARLA_RUN_AS_UID:-1000}" \
      CARLA_RUN_AS_GID="${CARLA_RUN_AS_GID:-1000}" \
      CARLA_RUN_AS_GROUPS="${CARLA_RUN_AS_GROUPS:-44,109}" \
      CARLA_LOG="$carla_log" \
      COLLECT_CONFIG="$COLLECT_CONFIG" \
      FORCE_RESTART_CARLA=1 \
      bash configs/collect_autocarla_server.sh >> "$RUN_LOG" 2>&1
    rc=$?
    set -e

    kill_carla_on_port "$PORT"

    if [[ "$rc" == "0" ]] && checkpoint_is_successful "$checkpoint"; then
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
frame_count=$(find "$OUTPUT_ROOT/data" -path '*/measurements/*.json.gz' 2>/dev/null | wc -l | tr -d ' ')
log "supervised collection finished done=${done_count} failed=${fail_count} frames=${frame_count}"
