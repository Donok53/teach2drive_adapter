#!/usr/bin/env bash

set -euo pipefail

export HOME="${HOME:-/home/jovyan}"

ADAPTER_ROOT="${ADAPTER_ROOT:-${HOME}/teach2drive/workspace/teach2drive_adapter}"
CARLA_ROOT="${CARLA_ROOT:-${HOME}/dataset/byeongjae/carla-simulator}"
GARAGE_ROOT="${GARAGE_ROOT:-${HOME}/teach2drive/workspace/carla_garage}"
TEAM_CONFIG="${TEAM_CONFIG:-${HOME}/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns}"
CHECKPOINT="${CHECKPOINT:-${HOME}/dataset/byeongjae/runs/tfpp_tesla_town13_task_feature_adapter_sgpu/train_front_triplet_shifted_task_feature_adapter_sgpu/best_model.pt}"
AGENT_PATH="${AGENT_PATH:-${ADAPTER_ROOT}/scripts/tfpp_feature_then_fusion_peft_adapter_sensor_rig_agent.py}"

ROUTE_XML="${ROUTE_XML:-${ADAPTER_ROOT}/runs/leaderboard_tfpp_missions/routes_validation_01/mission_001_src1_s001_SignalizedJunctionRightTurn_SignalizedJunctionRightTurn_1.xml}"
RUN_DIR="${RUN_DIR:-${HOME}/dataset/byeongjae/runs/eval_task_feature_sgpu_mission1_t180}"
LOG_DIR="${LOG_DIR:-${HOME}/teach2drive/logs}"
PORT="${PORT:-2011}"
TM_PORT="${TM_PORT:-8011}"
CARLA_GRAPHICS_ADAPTER="${CARLA_GRAPHICS_ADAPTER:-7}"
MISSION_TIMEOUT_SEC="${MISSION_TIMEOUT_SEC:-180}"

MISSION="mission_001_src1_s001_SignalizedJunctionRightTurn_SignalizedJunctionRightTurn_1"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/eval_task_feature_sgpu_mission1_t180.log}"
CARLA_LOG="${RUN_DIR}/carla_${PORT}.log"
MISSION_LOG="${RUN_DIR}/logs/${MISSION}.log"
CHECKPOINT_OUTPUT="${RUN_DIR}/results/${MISSION}.json"
VIDEO_OUTPUT="${RUN_DIR}/videos/${MISSION}_input_hq.mp4"
SUMMARY_TSV="${RUN_DIR}/summary.tsv"

mkdir -p "${LOG_DIR}" "${RUN_DIR}/logs" "${RUN_DIR}/results" "${RUN_DIR}/videos"

if [[ ! -f "${ROUTE_XML}" ]]; then
  echo "missing route xml: ${ROUTE_XML}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

export PATH="${HOME}/.venv/torch2.1.2-py3.10-cuda11.8/bin:${PATH}"
export PYTHONPATH="${ADAPTER_ROOT}:${CARLA_ROOT}/PythonAPI/carla:${GARAGE_ROOT}/leaderboard:${GARAGE_ROOT}/scenario_runner:${GARAGE_ROOT}/team_code:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "${RUN_LOG}"
}

stop_carla() {
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  mapfile -t pids < <(
    ps -eo pid=,comm=,args= | awk -v port="-carla-rpc-port=${PORT}" '
      $2 ~ /^CarlaUE4/ && index($0, port) > 0 {print $1}
    '
  )
  if (( ${#pids[@]} > 0 )); then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 2
    kill -KILL "${pids[@]}" 2>/dev/null || true
  fi
}

carla_ready() {
  PORT="${PORT}" timeout 8s python - <<'PY' >/dev/null 2>&1
import os
import carla

client = carla.Client("127.0.0.1", int(os.environ["PORT"]))
client.set_timeout(4.0)
client.get_world()
PY
}

cleanup() {
  stop_carla
}
trap cleanup EXIT

log "eval mission1 start port=${PORT} tm_port=${TM_PORT} gpu=${CARLA_GRAPHICS_ADAPTER}"
log "route=${ROUTE_XML}"
log "checkpoint=${CHECKPOINT}"

stop_carla
pushd "${CARLA_ROOT}" >/dev/null
setsid ./CarlaUE4.sh \
  -RenderOffScreen \
  -nosound \
  -quality-level=Low \
  "-graphicsadapter=${CARLA_GRAPHICS_ADAPTER}" \
  "-carla-rpc-port=${PORT}" \
  -stdout \
  -FullStdOutLogOutput \
  > "${CARLA_LOG}" 2>&1 &
CARLA_PID=$!
popd >/dev/null
log "CARLA_PID=${CARLA_PID} log=${CARLA_LOG}"

deadline=$((SECONDS + 150))
until carla_ready; do
  if (( SECONDS >= deadline )); then
    log "CARLA did not become ready"
    tail -n 80 "${CARLA_LOG}" || true
    exit 2
  fi
  sleep 2
done
log "CARLA ready"

rm -f "${CHECKPOINT_OUTPUT}" "${VIDEO_OUTPUT}" "${MISSION_LOG}"
cd "${ADAPTER_ROOT}"

set +e
timeout --signal=INT -k 30s "${MISSION_TIMEOUT_SEC}s" \
  env \
    ADAPTER_ROOT="${ADAPTER_ROOT}" \
    CARLA_ROOT="${CARLA_ROOT}" \
    GARAGE_ROOT="${GARAGE_ROOT}" \
    TEAM_CONFIG="${TEAM_CONFIG}" \
    ROUTE_XML="${ROUTE_XML}" \
    RUN_DIR="${RUN_DIR}" \
    PORT="${PORT}" \
    TM_PORT="${TM_PORT}" \
    VIDEO_OUTPUT="${VIDEO_OUTPUT}" \
    CHECKPOINT_OUTPUT="${CHECKPOINT_OUTPUT}" \
    RECORDER_LOG="${RUN_DIR}/logs/${MISSION}_recorder.log" \
    RECORD_VIDEO=0 \
    VIDEO_VIEW=input_hq \
    VIDEO_CODEC=mp4v \
    VIDEO_NOTION_COMPAT=0 \
    EGO_VEHICLE_MODEL=vehicle.tesla.model3 \
    DEBUG=1 \
    AGENT_PATH="${AGENT_PATH}" \
    TFPP_SENSOR_RIG=front_triplet_shifted \
    TFPP_SENSOR_CAMERA=front \
    TFPP_SENSOR_LIDAR=top \
    TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT="${CHECKPOINT}" \
    TFPP_FEATURE_ADAPTER_BLEND=1.0 \
    TFPP_STAGE_FEATURE_ADAPTER_BLEND=1.0 \
    TFPP_FUSION_ADAPTER_BLEND=1.0 \
    TFPP_AGENT_RECORD_VIDEO=1 \
    TFPP_RECORD_SENSOR_ID=rgb_front \
    TFPP_RECORD_FPS=20 \
    TFPP_RECORD_CODEC=mp4v \
    TFPP_RECORD_EVERY_N=1 \
    bash "${ADAPTER_ROOT}/scripts/run_tfpp_town13_with_video.sh" \
  > "${MISSION_LOG}" 2>&1
exit_code=$?
set -e

python - "${CHECKPOINT_OUTPUT}" "${exit_code}" "${SUMMARY_TSV}" "${MISSION}" "${VIDEO_OUTPUT}" <<'PY' | tee -a "${RUN_LOG}"
import csv
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
exit_code = int(sys.argv[2])
summary = Path(sys.argv[3])
mission = sys.argv[4]
video = sys.argv[5]

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

summary.parent.mkdir(parents=True, exist_ok=True)
write_header = not summary.exists()
with summary.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, delimiter="\t")
    if write_header:
        writer.writerow(["index", "mission", "outcome", "exit_code", "status", "score_route", "score_penalty", "score_composed", "num_infractions", "checkpoint", "video"])
    writer.writerow([1, mission, outcome, exit_code, status, f"{score_route:.3f}", f"{score_penalty:.3f}", f"{score_composed:.3f}", num_infractions, str(checkpoint), video])

print("=== mission1 summary")
print(f"outcome={outcome} exit_code={exit_code} status={status}")
print(f"score_route={score_route:.3f} score_penalty={score_penalty:.3f} score_composed={score_composed:.3f} infractions={num_infractions}")
print(f"summary_tsv={summary}")
print(f"mission_log={summary.parent / 'logs' / (mission + '.log')}")
PY

log "eval mission1 done exit_code=${exit_code}"
exit "${exit_code}"
