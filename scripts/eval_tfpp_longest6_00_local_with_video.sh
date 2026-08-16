#!/usr/bin/env bash
set -euo pipefail

CARLA_ROOT="${CARLA_ROOT:-/home/byeongjae/carla-simulator}"
GARAGE_ROOT="${GARAGE_ROOT:-/home/byeongjae/code/carla_garage}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}"
TEAM_CONFIG="${TEAM_CONFIG:-/home/byeongjae/code/checkpoints/transfuserpp/pretrained_models/all_towns}"
AGENT_PATH="${AGENT_PATH:-${GARAGE_ROOT}/team_code/sensor_agent.py}"
ROUTE_XML="${ROUTE_XML:-${GARAGE_ROOT}/leaderboard/data/longest6_split/longest6_00.xml}"
RUN_DIR="${RUN_DIR:-${ADAPTER_ROOT}/runs/eval_tfpp_baseline_longest6_00_local}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
PORT="${PORT:-2017}"
TM_PORT="${TM_PORT:-8017}"
SEED="${SEED:-42}"
CARLA_GRAPHICS_ADAPTER="${CARLA_GRAPHICS_ADAPTER:-1}"
TORCH_CUDA_VISIBLE_DEVICES="${TORCH_CUDA_VISIBLE_DEVICES:-${CARLA_GRAPHICS_ADAPTER}}"
EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC:-5400}"
EGO_VEHICLE_MODEL="${EGO_VEHICLE_MODEL:-vehicle.lincoln.mkz_2020}"

RUN_LOG="${RUN_LOG:-${LOG_DIR}/eval_tfpp_baseline_longest6_00.log}"
CARLA_LOG="${CARLA_LOG:-${LOG_DIR}/carla_port${PORT}.log}"
EVAL_LOG="${EVAL_LOG:-${LOG_DIR}/leaderboard_longest6_00.log}"
CHECKPOINT_OUTPUT="${CHECKPOINT_OUTPUT:-${RUN_DIR}/results/longest6_00.json}"
VIDEO_OUTPUT="${VIDEO_OUTPUT:-${RUN_DIR}/videos/longest6_00_topdown.mp4}"
RECORDER_LOG="${RECORDER_LOG:-${LOG_DIR}/recorder_longest6_00.log}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}/results" "${RUN_DIR}/videos"
: > "${RUN_LOG}"

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
    log "stopping stale CARLA on port ${PORT}: ${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 3
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
  stop_carla || true
}
trap cleanup EXIT

if [[ ! -f "${CARLA_ROOT}/CarlaUE4.sh" ]]; then
  log "missing CARLA executable: ${CARLA_ROOT}/CarlaUE4.sh"
  exit 2
fi
if [[ ! -f "${ROUTE_XML}" ]]; then
  log "missing route xml: ${ROUTE_XML}"
  exit 2
fi
if [[ ! -f "${AGENT_PATH}" ]]; then
  log "missing agent: ${AGENT_PATH}"
  exit 2
fi
if [[ ! -d "${TEAM_CONFIG}" && ! -f "${TEAM_CONFIG}" ]]; then
  log "missing team config: ${TEAM_CONFIG}"
  exit 2
fi

export PATH="/home/byeongjae/miniconda3/envs/mos4d/bin:${PATH}"
export PYTHONPATH="${ADAPTER_ROOT}:${CARLA_ROOT}/PythonAPI/carla:${GARAGE_ROOT}/leaderboard:${GARAGE_ROOT}/scenario_runner:${GARAGE_ROOT}/team_code:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"

log "TF++ baseline Longest6 local eval start"
log "route=${ROUTE_XML}"
log "run_dir=${RUN_DIR}"
log "ego_vehicle=${EGO_VEHICLE_MODEL}"
log "port=${PORT} tm_port=${TM_PORT} carla_gpu=${CARLA_GRAPHICS_ADAPTER} torch_cuda_visible_devices=${TORCH_CUDA_VISIBLE_DEVICES}"

rm -f "${CHECKPOINT_OUTPUT}" "${VIDEO_OUTPUT}" "${EVAL_LOG}" "${RECORDER_LOG}"
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

deadline=$((SECONDS + 180))
until carla_ready; do
  if (( SECONDS >= deadline )); then
    log "CARLA did not become ready"
    tail -80 "${CARLA_LOG}" >> "${RUN_LOG}" 2>/dev/null || true
    exit 2
  fi
  sleep 2
done
log "CARLA ready"

set +e
timeout --signal=INT -k 60s "${EVAL_TIMEOUT_SEC}s" \
  env \
    CUDA_VISIBLE_DEVICES="${TORCH_CUDA_VISIBLE_DEVICES}" \
    CARLA_ROOT="${CARLA_ROOT}" \
    GARAGE_ROOT="${GARAGE_ROOT}" \
    ADAPTER_ROOT="${ADAPTER_ROOT}" \
    TEAM_CONFIG="${TEAM_CONFIG}" \
    AGENT_PATH="${AGENT_PATH}" \
    ROUTE_XML="${ROUTE_XML}" \
    RUN_DIR="${RUN_DIR}" \
    PORT="${PORT}" \
    TM_PORT="${TM_PORT}" \
    SEED="${SEED}" \
    DEBUG=1 \
    RECORD_VIDEO=1 \
    VIDEO_VIEW=topdown \
    VIDEO_WIDTH=1280 \
    VIDEO_HEIGHT=720 \
    VIDEO_FPS=20 \
    VIDEO_CODEC=mp4v \
    VIDEO_NOTION_COMPAT=1 \
    VIDEO_OUTPUT="${VIDEO_OUTPUT}" \
    CHECKPOINT_OUTPUT="${CHECKPOINT_OUTPUT}" \
    RECORDER_LOG="${RECORDER_LOG}" \
    EGO_VEHICLE_MODEL="${EGO_VEHICLE_MODEL}" \
    bash "${ADAPTER_ROOT}/scripts/run_tfpp_town13_with_video.sh" \
  > "${EVAL_LOG}" 2>&1
exit_code=$?
set -e

log "leaderboard exit_code=${exit_code} eval_log=${EVAL_LOG}"
if [[ -f "${CHECKPOINT_OUTPUT}" ]]; then
  python - "${CHECKPOINT_OUTPUT}" <<'PY' | tee -a "${RUN_LOG}" || true
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
records = data.get("_checkpoint", {}).get("records", [])
if records:
    record = records[-1]
    print("status=", record.get("status"))
    print("scores=", record.get("scores"))
    print("num_infractions=", record.get("num_infractions"))
else:
    print("no records in result")
print("entry_status=", data.get("entry_status"))
PY
else
  log "no result json written"
fi
if [[ -s "${VIDEO_OUTPUT}" ]]; then
  log "video=${VIDEO_OUTPUT}"
else
  log "video not written yet: ${VIDEO_OUTPUT}"
fi

exit "${exit_code}"
