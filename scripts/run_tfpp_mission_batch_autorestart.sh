#!/usr/bin/env bash

set -u

ADAPTER_ROOT="${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}"
CARLA_ROOT="${CARLA_ROOT:-/home/byeongjae/carla-simulator}"
MISSION_DIR="${MISSION_DIR:-${ADAPTER_ROOT}/runs/leaderboard_tfpp_missions/routes_validation_01}"
MISSION_LIST="${MISSION_LIST:-${MISSION_DIR}/mission_routes.txt}"
RUN_DIR="${RUN_DIR:-${MISSION_DIR}/tfpp_runs_autorestart}"
SUMMARY_TSV="${SUMMARY_TSV:-${RUN_DIR}/summary.tsv}"
ATTEMPT_TSV="${ATTEMPT_TSV:-${SUMMARY_TSV%.tsv}_attempts.tsv}"
PORT="${PORT:-2000}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-0}"
MAX_RESTARTS="${MAX_RESTARTS:-12}"
CARLA_WAIT_SEC="${CARLA_WAIT_SEC:-120}"
CARLA_QUALITY="${CARLA_QUALITY:-Low}"
CARLA_EXTRA_ARGS="${CARLA_EXTRA_ARGS:-}"
KEEP_CARLA="${KEEP_CARLA:-0}"
STOP_ON_TIMEOUT="${STOP_ON_TIMEOUT:-0}"
STOP_ON_INVALID="${STOP_ON_INVALID:-0}"
INVALID_RETRY_LIMIT="${INVALID_RETRY_LIMIT:-0}"
FRESH_CARLA_ON_START="${FRESH_CARLA_ON_START:-0}"
CARLA_WATCHDOG="${CARLA_WATCHDOG:-1}"
CARLA_WATCHDOG_INTERVAL="${CARLA_WATCHDOG_INTERVAL:-5}"
CARLA_WATCHDOG_MISSES="${CARLA_WATCHDOG_MISSES:-3}"
MISSION_HEARTBEAT_SEC="${MISSION_HEARTBEAT_SEC:-60}"

export PATH="/home/byeongjae/miniconda3/envs/mos4d/bin:${PATH}"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH:-}"

mkdir -p "${RUN_DIR}/carla_logs"

CARLA_PID=""

mission_count() {
  awk 'NF {count += 1} END {print count + 0}' "${MISSION_LIST}"
}

max_recorded_index() {
  python - "${SUMMARY_TSV}" "$((START_INDEX - 1))" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
default = int(sys.argv[2])
if not path.exists():
    print(default)
    raise SystemExit
best = default
with path.open("r", encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        try:
            best = max(best, int(row.get("index", default)))
        except Exception:
            pass
print(best)
PY
}

carla_ready_once() {
  PORT="${PORT}" timeout 6s python - <<'PY' >/dev/null 2>&1
import os
import carla

client = carla.Client("127.0.0.1", int(os.environ["PORT"]))
client.set_timeout(3.0)
client.get_world()
PY
}

wait_for_carla() {
  local deadline=$((SECONDS + CARLA_WAIT_SEC))
  while (( SECONDS < deadline )); do
    if carla_ready_once; then
      return 0
    fi
    sleep 2
  done
  return 1
}

start_carla() {
  if [[ "${FRESH_CARLA_ON_START}" != "1" ]] && carla_ready_once; then
    echo "CARLA already reachable on port ${PORT}"
    return 0
  fi

  local restart_id="$1"
  local route_tag="${current_start:-start}"
  local timestamp
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  local log="${RUN_DIR}/carla_logs/carla_route_${route_tag}_restart_${restart_id}_${timestamp}.log"
  local cmd=(./CarlaUE4.sh -RenderOffScreen -nosound -quality-level="${CARLA_QUALITY}" -carla-rpc-port="${PORT}")
  if [[ -n "${CARLA_EXTRA_ARGS}" ]]; then
    # Intentionally split shell-style extra args, e.g. CARLA_EXTRA_ARGS="-windowed -ResX=960 -ResY=540".
    # shellcheck disable=SC2206
    local extra=(${CARLA_EXTRA_ARGS})
    cmd+=("${extra[@]}")
  fi

  echo "Starting CARLA restart=${restart_id} log=${log}"
  pushd "${CARLA_ROOT}" >/dev/null || return 1
  setsid "${cmd[@]}" > "${log}" 2>&1 &
  CARLA_PID=$!
  CARLA_LOG="${log}"
  popd >/dev/null || true

  if wait_for_carla; then
    return 0
  fi

  echo "CARLA did not become ready; see ${log}" >&2
  stop_carla
  return 1
}

stop_carla() {
  if [[ -n "${CARLA_PID}" ]]; then
    kill -TERM "-${CARLA_PID}" 2>/dev/null || kill -TERM "${CARLA_PID}" 2>/dev/null || true
    sleep 3
    kill -KILL "-${CARLA_PID}" 2>/dev/null || kill -KILL "${CARLA_PID}" 2>/dev/null || true
    wait "${CARLA_PID}" 2>/dev/null || true
    CARLA_PID=""
  fi
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  mapfile -t carla_pids < <(
    ps -eo pid=,comm=,args= | awk -v port="-carla-rpc-port=${PORT}" '
      ($2 ~ /CarlaUE4/ || $0 ~ /CarlaUE4-Linux-Shipping/) && index($0, port) > 0 {print $1}
    '
  )
  if (( ${#carla_pids[@]} > 0 )); then
    kill -TERM "${carla_pids[@]}" 2>/dev/null || true
  fi
  sleep 1
  if (( ${#carla_pids[@]} > 0 )); then
    kill -KILL "${carla_pids[@]}" 2>/dev/null || true
  fi
}

cleanup() {
  if [[ "${KEEP_CARLA}" != "1" ]]; then
    stop_carla
  fi
}
trap cleanup EXIT

total_missions="$(mission_count)"
if (( total_missions <= 0 )); then
  echo "No missions found in ${MISSION_LIST}" >&2
  exit 1
fi

target_end=$((total_missions - 1))
if (( LIMIT > 0 )); then
  requested_end=$((START_INDEX + LIMIT - 1))
  if (( requested_end < target_end )); then
    target_end="${requested_end}"
  fi
fi

current_start="${START_INDEX}"
restart_count=0

if [[ "${FRESH_CARLA_ON_START}" == "1" ]]; then
  stop_carla
fi

while (( current_start <= target_end )); do
  if ! carla_ready_once; then
    stop_carla
    if (( restart_count >= MAX_RESTARTS )); then
      echo "Exceeded MAX_RESTARTS=${MAX_RESTARTS}; last START_INDEX=${current_start}" >&2
      exit 2
    fi
    restart_count=$((restart_count + 1))
    start_carla "${restart_count}" || {
      echo "Failed to start CARLA for START_INDEX=${current_start}" >&2
      exit 2
    }
  fi

  remaining=$((target_end - current_start + 1))
  before_max="$(max_recorded_index)"
  echo "=== batch START_INDEX=${current_start} LIMIT=${remaining} target_end=${target_end} restarts=${restart_count}"

  START_INDEX="${current_start}" \
  LIMIT="${remaining}" \
  RUN_DIR="${RUN_DIR}" \
  SUMMARY_TSV="${SUMMARY_TSV}" \
  ATTEMPT_TSV="${ATTEMPT_TSV}" \
  PORT="${PORT}" \
  STOP_ON_TIMEOUT="${STOP_ON_TIMEOUT}" \
  STOP_ON_INVALID="${STOP_ON_INVALID}" \
  INVALID_RETRY_LIMIT="${INVALID_RETRY_LIMIT}" \
  CARLA_WATCHDOG="${CARLA_WATCHDOG}" \
  CARLA_WATCHDOG_INTERVAL="${CARLA_WATCHDOG_INTERVAL}" \
  CARLA_WATCHDOG_MISSES="${CARLA_WATCHDOG_MISSES}" \
  MISSION_HEARTBEAT_SEC="${MISSION_HEARTBEAT_SEC}" \
  ACTIVE_CARLA_PID="${CARLA_PID}" \
  CARLA_LOG="${CARLA_LOG:-}" \
  bash "${ADAPTER_ROOT}/scripts/run_tfpp_mission_batch.sh"
  status=$?

  after_max="$(max_recorded_index)"
  if (( after_max >= current_start )); then
    current_start=$((after_max + 1))
  elif (( after_max > before_max )); then
    current_start=$((after_max + 1))
  else
    echo "No mission progress recorded for START_INDEX=${current_start}" >&2
  fi

  if (( current_start > target_end )); then
    break
  fi

  if ! carla_ready_once || (( status != 0 )); then
    echo "CARLA/batch ended with status=${status}; restarting before START_INDEX=${current_start}" >&2
    stop_carla
    continue
  fi

  # The inner batch can return 0 after STOP_ON_INVALID/STOP_ON_TIMEOUT. Continue
  # from the next unrecorded mission while the simulator is still alive.
done

echo "=== autorestart complete"
python - "${SUMMARY_TSV}" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open("r", encoding="utf-8"), delimiter="\t")) if path.exists() else []
counts = Counter(row["outcome"] for row in rows)
print(f"summary_tsv={path}")
print(f"total={len(rows)} pass={counts['PASS']} fail={counts['FAIL']} invalid={counts['INVALID']}")
if rows:
    print(f"mean_route={sum(float(row['score_route']) for row in rows) / len(rows):.2f}")
    print(f"mean_composed={sum(float(row['score_composed']) for row in rows) / len(rows):.2f}")
PY
