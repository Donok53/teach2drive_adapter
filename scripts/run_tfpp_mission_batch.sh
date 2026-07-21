#!/usr/bin/env bash

set -u

ADAPTER_ROOT="${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}"
MISSION_DIR="${MISSION_DIR:-${ADAPTER_ROOT}/runs/leaderboard_tfpp_missions/routes_validation_01}"
MISSION_LIST="${MISSION_LIST:-${MISSION_DIR}/mission_routes.txt}"
RUN_DIR="${RUN_DIR:-${MISSION_DIR}/tfpp_runs}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-0}"
EGO_VEHICLE_MODEL="${EGO_VEHICLE_MODEL:-vehicle.tesla.model3}"
VIDEO_VIEW="${VIDEO_VIEW:-topdown}"
VIDEO_EXT="${VIDEO_EXT:-mp4}"
RECORD_VIDEO="${RECORD_VIDEO:-1}"
VIDEO_RECORD_INDICES="${VIDEO_RECORD_INDICES:-}"
VIDEO_WIDTH="${VIDEO_WIDTH:-1280}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-720}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_FOV="${VIDEO_FOV:-90}"
VIDEO_CODEC="${VIDEO_CODEC:-mp4v}"
VIDEO_SENSOR_TICK="${VIDEO_SENSOR_TICK:-0.05}"
VIDEO_PIP_VIEW="${VIDEO_PIP_VIEW:-}"
VIDEO_PIP_SCALE="${VIDEO_PIP_SCALE:-0.30}"
VIDEO_PIP_MARGIN="${VIDEO_PIP_MARGIN:-24}"
VIDEO_PIP_BORDER="${VIDEO_PIP_BORDER:-2}"
VIDEO_NOTION_COMPAT="${VIDEO_NOTION_COMPAT:-1}"
VIDEO_NOTION_CRF="${VIDEO_NOTION_CRF:-23}"
VIDEO_NOTION_PRESET="${VIDEO_NOTION_PRESET:-medium}"
DEBUG="${DEBUG:-1}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
AGENT_PATH="${AGENT_PATH:-}"
TFPP_SENSOR_RIG="${TFPP_SENSOR_RIG:-}"
TFPP_SENSOR_RIG_JSON="${TFPP_SENSOR_RIG_JSON:-}"
TFPP_SENSOR_CAMERA="${TFPP_SENSOR_CAMERA:-front}"
TFPP_SENSOR_LIDAR="${TFPP_SENSOR_LIDAR:-top}"
TFPP_ADAPTER_CHECKPOINT="${TFPP_ADAPTER_CHECKPOINT:-}"
TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT="${TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT:-}"
TFPP_FEATURE_ADAPTER_BLEND="${TFPP_FEATURE_ADAPTER_BLEND:-}"
TFPP_STAGE_FEATURE_ADAPTER_BLEND="${TFPP_STAGE_FEATURE_ADAPTER_BLEND:-}"
TFPP_FUSION_ADAPTER_BLEND="${TFPP_FUSION_ADAPTER_BLEND:-}"
TFPP_ADAPTER_CAMERAS="${TFPP_ADAPTER_CAMERAS:-}"
TFPP_ADAPTER_SPEED_MODE="${TFPP_ADAPTER_SPEED_MODE:-}"
TFPP_ADAPTER_SPEED_BLEND="${TFPP_ADAPTER_SPEED_BLEND:-}"
TFPP_ADAPTER_USE_STOP="${TFPP_ADAPTER_USE_STOP:-}"
TFPP_ADAPTER_STOP_THRESHOLD="${TFPP_ADAPTER_STOP_THRESHOLD:-}"
TFPP_ADAPTER_CONTROL_MODE="${TFPP_ADAPTER_CONTROL_MODE:-}"
TFPP_ADAPTER_DEBUG_EVERY="${TFPP_ADAPTER_DEBUG_EVERY:-}"
TFPP_AGENT_RECORD_VIDEO="${TFPP_AGENT_RECORD_VIDEO:-}"
TFPP_RECORD_SENSOR_ID="${TFPP_RECORD_SENSOR_ID:-}"
TFPP_RECORD_FPS="${TFPP_RECORD_FPS:-}"
TFPP_RECORD_CODEC="${TFPP_RECORD_CODEC:-}"
TFPP_RECORD_EVERY_N="${TFPP_RECORD_EVERY_N:-}"
TFPP_RECORD_SCALE="${TFPP_RECORD_SCALE:-}"
SUMMARY_TSV="${SUMMARY_TSV:-${RUN_DIR}/summary.tsv}"
ATTEMPT_TSV="${ATTEMPT_TSV:-${SUMMARY_TSV%.tsv}_attempts.tsv}"
MISSION_TIMEOUT_SEC="${MISSION_TIMEOUT_SEC:-420}"
STOP_ON_TIMEOUT="${STOP_ON_TIMEOUT:-0}"
STOP_ON_INVALID="${STOP_ON_INVALID:-1}"
INVALID_RETRY_LIMIT="${INVALID_RETRY_LIMIT:-0}"
CLEANUP_AFTER_MISSION="${CLEANUP_AFTER_MISSION:-1}"
CARLA_WATCHDOG="${CARLA_WATCHDOG:-1}"
CARLA_WATCHDOG_INTERVAL="${CARLA_WATCHDOG_INTERVAL:-5}"
CARLA_WATCHDOG_MISSES="${CARLA_WATCHDOG_MISSES:-3}"
MISSION_HEARTBEAT_SEC="${MISSION_HEARTBEAT_SEC:-60}"
ACTIVE_CARLA_PID="${ACTIVE_CARLA_PID:-}"
CARLA_LOG="${CARLA_LOG:-}"
IS_BENCH2DRIVE="${IS_BENCH2DRIVE:-}"

if [[ ! "${DEBUG}" =~ ^-?[0-9]+$ ]]; then
  echo "WARNING: DEBUG=${DEBUG} is not an integer; using DEBUG=1" >&2
  DEBUG=1
fi

if [[ "${VIDEO_NOTION_COMPAT}" == "1" ]]; then
  VIDEO_EXT="mp4"
  VIDEO_CODEC="mp4v"
  if [[ -z "${TFPP_RECORD_CODEC}" || "${TFPP_RECORD_CODEC}" == "MJPG" ]]; then
    TFPP_RECORD_CODEC="mp4v"
  fi
fi

mkdir -p "${RUN_DIR}/videos" "${RUN_DIR}/results" "${RUN_DIR}/logs"

if (( START_INDEX > 0 )) && [[ -f "${SUMMARY_TSV}" ]]; then
  :
else
  printf "index\tmission\toutcome\texit_code\tstatus\tscore_route\tscore_penalty\tscore_composed\tnum_infractions\tcheckpoint\tvideo\n" > "${SUMMARY_TSV}"
fi
if [[ ! -f "${ATTEMPT_TSV}" ]]; then
  printf "index\tmission\tattempt_outcome\texit_code\tstatus\tscore_route\tscore_penalty\tscore_composed\tnum_infractions\tcheckpoint\tvideo\n" > "${ATTEMPT_TSV}"
fi

cleanup_carla_world() {
  [[ "${CLEANUP_AFTER_MISSION}" == "1" ]] || return 0
  timeout 20s env PORT="${PORT}" TM_PORT="${TM_PORT}" python - <<'PY' >/dev/null 2>&1 || true
import os
import time

import carla

client = carla.Client("127.0.0.1", int(os.environ["PORT"]))
client.set_timeout(5.0)
world = client.get_world()

try:
    tm = client.get_trafficmanager(int(os.environ["TM_PORT"]))
    tm.set_synchronous_mode(False)
except Exception:
    pass

try:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
except Exception:
    pass

try:
    actors = world.get_actors()
    destroy = []
    for pattern in ("sensor.*", "vehicle.*", "walker.*", "controller.ai.walker"):
        destroy.extend(carla.command.DestroyActor(actor) for actor in actors.filter(pattern))
    if destroy:
        client.apply_batch_sync(destroy, False)
except Exception:
    pass

time.sleep(1.0)
PY
}

carla_tcp_reachable() {
  timeout 3s bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null
}

carla_process_alive() {
  [[ -z "${ACTIVE_CARLA_PID}" ]] || kill -0 "${ACTIVE_CARLA_PID}" 2>/dev/null
}

append_carla_crash_context() {
  local log="$1"
  [[ -n "${CARLA_LOG}" && -f "${CARLA_LOG}" ]] || return 0
  {
    echo
    echo "=== CARLA log tail: ${CARLA_LOG}"
    tail -n 80 "${CARLA_LOG}" || true
  } >> "${log}" 2>/dev/null || true
}

interrupt_process_group() {
  local pid="$1"
  local sid=""
  sid="$(ps -o sid= -p "${pid}" 2>/dev/null | awk '{print $1}')"
  kill -INT "-${pid}" 2>/dev/null || kill -INT "${pid}" 2>/dev/null || true
  sleep 8
  kill -TERM "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  if [[ -n "${sid}" ]]; then
    pkill -TERM -s "${sid}" 2>/dev/null || true
  fi
  sleep 4
  kill -KILL "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  if [[ -n "${sid}" ]]; then
    pkill -KILL -s "${sid}" 2>/dev/null || true
  fi
}

run_mission_with_watchdog() {
  local log="$1"
  shift
  local mission_pid
  local misses=0
  local start_ts
  local last_heartbeat
  local now
  local watchdog_reason=""

  start_ts="$(date +%s)"
  last_heartbeat="${start_ts}"
  setsid "$@" > "${log}" 2>&1 < /dev/null &
  mission_pid=$!

  while kill -0 "${mission_pid}" 2>/dev/null; do
    sleep "${CARLA_WATCHDOG_INTERVAL}"
    now="$(date +%s)"
    if (( MISSION_HEARTBEAT_SEC > 0 && now - last_heartbeat >= MISSION_HEARTBEAT_SEC )); then
      echo "Mission heartbeat pid=${mission_pid} elapsed=$((now - start_ts))s carla_tcp=$(carla_tcp_reachable && echo ok || echo down)" >&2
      last_heartbeat="${now}"
    fi
    [[ "${CARLA_WATCHDOG}" == "1" ]] || continue
    if carla_process_alive && carla_tcp_reachable; then
      misses=0
      continue
    fi
    misses=$((misses + 1))
    echo "CARLA watchdog miss ${misses}/${CARLA_WATCHDOG_MISSES} on port ${PORT} for pid=${mission_pid}" >&2
    echo "CARLA watchdog miss ${misses}/${CARLA_WATCHDOG_MISSES} on port ${PORT}" >> "${log}" 2>/dev/null || true
    if (( misses >= CARLA_WATCHDOG_MISSES )); then
      watchdog_reason="CARLA unreachable/dead for $((misses * CARLA_WATCHDOG_INTERVAL))s"
      echo "CARLA watchdog stopping mission pid=${mission_pid}: ${watchdog_reason}" >&2
      echo "CARLA watchdog stopping mission: ${watchdog_reason}" >> "${log}" 2>/dev/null || true
      append_carla_crash_context "${log}"
      interrupt_process_group "${mission_pid}"
      break
    fi
  done

  wait "${mission_pid}"
  local status=$?
  if [[ -n "${watchdog_reason}" ]]; then
    return 134
  fi
  return "${status}"
}

should_record_video() {
  local idx="$1"
  [[ "${RECORD_VIDEO}" == "1" ]] || return 1
  if [[ -z "${VIDEO_RECORD_INDICES}" ]]; then
    return 0
  fi
  local item
  IFS=',' read -ra _video_indices <<< "${VIDEO_RECORD_INDICES}"
  for item in "${_video_indices[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ "${item}" == "${idx}" ]]; then
      return 0
    fi
  done
  return 1
}

idx=-1
ran=0
while IFS= read -r ROUTE_XML || [[ -n "${ROUTE_XML}" ]]; do
  [[ -z "${ROUTE_XML}" ]] && continue
  idx=$((idx + 1))
  if (( idx < START_INDEX )); then
    continue
  fi
  if (( LIMIT > 0 && ran >= LIMIT )); then
    break
  fi

  mission="$(basename "${ROUTE_XML}" .xml)"
  checkpoint="${RUN_DIR}/results/${mission}.json"
  video_view_slug="${VIDEO_VIEW}"
  if [[ -n "${VIDEO_PIP_VIEW}" ]]; then
    video_view_slug="${VIDEO_VIEW}_pip_${VIDEO_PIP_VIEW}"
  fi
  video="${RUN_DIR}/videos/${mission}_${video_view_slug}.${VIDEO_EXT}"
  log="${RUN_DIR}/logs/${mission}.log"
  recorder_log="${RUN_DIR}/logs/${mission}_recorder.log"
  mission_record_video=0
  mission_video_notion_compat=0
  if should_record_video "${idx}"; then
    mission_record_video=1
    mission_video_notion_compat="${VIDEO_NOTION_COMPAT}"
  fi

  echo "=== mission ${idx}: ${mission} video=${mission_record_video}"
  if ! carla_tcp_reachable; then
    echo "CARLA server is not reachable on port ${PORT}; restart CARLA and resume with START_INDEX=${idx}" >&2
    exit 2
  fi
  rm -f "${checkpoint}" "${video}" "${log}" "${recorder_log}"
  mission_cmd=(
    timeout --signal=INT -k 30s "${MISSION_TIMEOUT_SEC}s"
    env
      ROUTE_XML="${ROUTE_XML}" \
      RUN_DIR="${RUN_DIR}" \
      PORT="${PORT}" \
      TM_PORT="${TM_PORT}" \
      VIDEO_OUTPUT="${video}" \
      CHECKPOINT_OUTPUT="${checkpoint}" \
      RECORDER_LOG="${recorder_log}" \
      VIDEO_VIEW="${VIDEO_VIEW}" \
      VIDEO_WIDTH="${VIDEO_WIDTH}" \
      VIDEO_HEIGHT="${VIDEO_HEIGHT}" \
      VIDEO_FPS="${VIDEO_FPS}" \
      VIDEO_FOV="${VIDEO_FOV}" \
      VIDEO_CODEC="${VIDEO_CODEC}" \
      VIDEO_SENSOR_TICK="${VIDEO_SENSOR_TICK}" \
      VIDEO_PIP_VIEW="${VIDEO_PIP_VIEW}" \
      VIDEO_PIP_SCALE="${VIDEO_PIP_SCALE}" \
      VIDEO_PIP_MARGIN="${VIDEO_PIP_MARGIN}" \
      VIDEO_PIP_BORDER="${VIDEO_PIP_BORDER}" \
      VIDEO_RECORD_INDICES="${VIDEO_RECORD_INDICES}" \
      VIDEO_NOTION_CRF="${VIDEO_NOTION_CRF}" \
      VIDEO_NOTION_PRESET="${VIDEO_NOTION_PRESET}" \
      RECORD_VIDEO="${mission_record_video}" \
      VIDEO_NOTION_COMPAT="${mission_video_notion_compat}" \
      EGO_VEHICLE_MODEL="${EGO_VEHICLE_MODEL}" \
      DEBUG="${DEBUG}" \
      IS_BENCH2DRIVE="${IS_BENCH2DRIVE}" \
      AGENT_PATH="${AGENT_PATH}" \
      TFPP_SENSOR_RIG="${TFPP_SENSOR_RIG}" \
      TFPP_SENSOR_RIG_JSON="${TFPP_SENSOR_RIG_JSON}" \
      TFPP_SENSOR_CAMERA="${TFPP_SENSOR_CAMERA}" \
      TFPP_SENSOR_LIDAR="${TFPP_SENSOR_LIDAR}" \
      TFPP_ADAPTER_CHECKPOINT="${TFPP_ADAPTER_CHECKPOINT}" \
      TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT="${TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT}" \
      TFPP_FEATURE_ADAPTER_BLEND="${TFPP_FEATURE_ADAPTER_BLEND}" \
      TFPP_STAGE_FEATURE_ADAPTER_BLEND="${TFPP_STAGE_FEATURE_ADAPTER_BLEND}" \
      TFPP_FUSION_ADAPTER_BLEND="${TFPP_FUSION_ADAPTER_BLEND}" \
      TFPP_ADAPTER_CAMERAS="${TFPP_ADAPTER_CAMERAS}" \
      TFPP_ADAPTER_SPEED_MODE="${TFPP_ADAPTER_SPEED_MODE}" \
      TFPP_ADAPTER_SPEED_BLEND="${TFPP_ADAPTER_SPEED_BLEND}" \
      TFPP_ADAPTER_USE_STOP="${TFPP_ADAPTER_USE_STOP}" \
      TFPP_ADAPTER_STOP_THRESHOLD="${TFPP_ADAPTER_STOP_THRESHOLD}" \
      TFPP_ADAPTER_CONTROL_MODE="${TFPP_ADAPTER_CONTROL_MODE}" \
      TFPP_ADAPTER_DEBUG_EVERY="${TFPP_ADAPTER_DEBUG_EVERY}" \
      TFPP_AGENT_RECORD_VIDEO="${TFPP_AGENT_RECORD_VIDEO}" \
      TFPP_RECORD_SENSOR_ID="${TFPP_RECORD_SENSOR_ID}" \
      TFPP_RECORD_FPS="${TFPP_RECORD_FPS}" \
      TFPP_RECORD_CODEC="${TFPP_RECORD_CODEC}" \
      TFPP_RECORD_EVERY_N="${TFPP_RECORD_EVERY_N}" \
      TFPP_RECORD_SCALE="${TFPP_RECORD_SCALE}" \
      bash "${ADAPTER_ROOT}/scripts/run_tfpp_town13_with_video.sh"
  )
  run_mission_with_watchdog "${log}" "${mission_cmd[@]}"
  exit_code=$?
  if [[ "${VIDEO_NOTION_COMPAT}" == "1" && -s "${video}" ]]; then
    python "${ADAPTER_ROOT}/scripts/convert_video_for_notion.py" \
      --input "${video}" \
      --crf "${VIDEO_NOTION_CRF}" \
      --preset "${VIDEO_NOTION_PRESET}" \
      >> "${log}" 2>&1 || true
  fi
  cleanup_carla_world

  parsed="$(
    python - "${checkpoint}" "${exit_code}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
exit_code = int(sys.argv[2])
status = "NO_RESULT"
score_route = 0.0
score_penalty = 0.0
score_composed = 0.0
num_infractions = -1
outcome = "INVALID"

if checkpoint.exists():
    try:
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        records = data.get("_checkpoint", {}).get("records", [])
        record = records[-1] if records else {}
        status = str(record.get("status", data.get("entry_status", "UNKNOWN")))
        scores = record.get("scores", {})
        score_route = float(scores.get("score_route", 0.0))
        score_penalty = float(scores.get("score_penalty", 0.0))
        score_composed = float(scores.get("score_composed", 0.0))
        num_infractions = int(record.get("num_infractions", 0))
        entry_status = str(data.get("entry_status", ""))
        if not records and exit_code == 124:
            status = "Failed - Mission wall-time timeout"
            outcome = "FAIL"
        elif "crash" in status.lower() or entry_status.lower() == "crashed" or exit_code not in (0, 1, 124):
            outcome = "INVALID"
        elif status in {"Perfect", "Completed"} and score_route >= 99.0 and score_penalty >= 1.0 and num_infractions == 0:
            outcome = "PASS"
        else:
            outcome = "FAIL"
    except Exception as exc:
        status = f"PARSE_ERROR:{exc}"
        outcome = "INVALID"
elif exit_code == 124:
    status = "Failed - Mission wall-time timeout"
    outcome = "FAIL"
else:
    outcome = "INVALID"

print("\t".join([
    outcome,
    str(exit_code),
    status.replace("\t", " "),
    f"{score_route:.3f}",
    f"{score_penalty:.3f}",
    f"{score_composed:.3f}",
    str(num_infractions),
]))
PY
  )"

  outcome="${parsed%%$'\t'*}"
  if [[ "${outcome}" == "INVALID" ]] && (( INVALID_RETRY_LIMIT > 0 )); then
    invalid_attempts="$(
      awk -F'\t' -v idx="${idx}" -v mission="${mission}" '
        NR > 1 && $1 == idx && $2 == mission && $3 == "INVALID" {count += 1}
        END {print count + 0}
      ' "${ATTEMPT_TSV}"
    )"
    printf "%s\t%s\t%s\t%s\t%s\n" "${idx}" "${mission}" "${parsed}" "${checkpoint}" "${video}" >> "${ATTEMPT_TSV}"
    if (( invalid_attempts < INVALID_RETRY_LIMIT )); then
      echo "Mission ${idx} ended INVALID; retry $((invalid_attempts + 1))/${INVALID_RETRY_LIMIT} after CARLA restart." >&2
      exit 3
    fi
    echo "Mission ${idx} stayed INVALID after $((invalid_attempts + 1)) attempt(s); recording final INVALID." >&2
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" "${idx}" "${mission}" "${parsed}" "${checkpoint}" "${video}" >> "${SUMMARY_TSV}"
  tail -n 1 "${SUMMARY_TSV}"
  ran=$((ran + 1))

  if (( exit_code == 124 )) && [[ "${STOP_ON_TIMEOUT}" == "1" ]]; then
    echo "Mission ${idx} hit wall-time timeout. Restart CARLA, then resume with START_INDEX=$((idx + 1))." >&2
    break
  fi
  if [[ "${outcome}" == "INVALID" && "${STOP_ON_INVALID}" == "1" ]]; then
    echo "Mission ${idx} ended INVALID. Restart CARLA, then resume with START_INDEX=$((idx + 1))." >&2
    break
  fi
done < "${MISSION_LIST}"

python - "${SUMMARY_TSV}" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open("r", encoding="utf-8"), delimiter="\t"))
counts = Counter(row["outcome"] for row in rows)
print("=== summary")
print(f"total={len(rows)} pass={counts['PASS']} fail={counts['FAIL']} invalid={counts['INVALID']}")
if rows:
    mean_route = sum(float(row["score_route"]) for row in rows) / len(rows)
    mean_score = sum(float(row["score_composed"]) for row in rows) / len(rows)
    print(f"mean_route={mean_route:.2f} mean_composed={mean_score:.2f}")
print(f"summary_tsv={path}")
PY
