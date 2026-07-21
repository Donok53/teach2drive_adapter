#!/usr/bin/env bash

CARLA_ROOT="${CARLA_ROOT:-/home/byeongjae/carla-simulator}"
GARAGE_ROOT="${GARAGE_ROOT:-/home/byeongjae/code/carla_garage}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}"
TEAM_CONFIG="${TEAM_CONFIG:-/home/byeongjae/code/checkpoints/transfuserpp/pretrained_models/all_towns}"
AGENT_PATH="${AGENT_PATH:-${GARAGE_ROOT}/team_code/sensor_agent.py}"
ROUTE_XML="${ROUTE_XML:-${GARAGE_ROOT}/leaderboard/data/routes_validation_split/routes_validation_00.xml}"
RUN_DIR="${RUN_DIR:-${ADAPTER_ROOT}/runs/leaderboard_tfpp_town13_validation}"
EGO_VEHICLE_MODEL="${EGO_VEHICLE_MODEL:-vehicle.tesla.model3}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
SEED="${SEED:-0}"
DEBUG="${DEBUG:-1}"
VIDEO_VIEW="${VIDEO_VIEW:-topdown}"
RECORD_VIDEO="${RECORD_VIDEO:-1}"
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
VIDEO_OUTPUT="${VIDEO_OUTPUT:-${RUN_DIR}/routes_validation_00_${VIDEO_VIEW}.mp4}"
CHECKPOINT_OUTPUT="${CHECKPOINT_OUTPUT:-${RUN_DIR}/result_routes_validation_00.json}"
RECORDER_LOG="${RECORDER_LOG:-${RUN_DIR}/recorder.log}"

if [[ "${VIDEO_NOTION_COMPAT}" == "1" && "${VIDEO_CODEC}" == "MJPG" ]]; then
  VIDEO_CODEC="mp4v"
fi

mkdir -p "${RUN_DIR}"

export PATH="/home/byeongjae/miniconda3/envs/mos4d/bin:${PATH}"
export CARLA_ROOT
export EGO_VEHICLE_MODEL
export WORK_DIR="${GARAGE_ROOT}"
export SCENARIO_RUNNER_ROOT="${GARAGE_ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${GARAGE_ROOT}/leaderboard"
export PYTHONPATH="${ADAPTER_ROOT}:${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${GARAGE_ROOT}/team_code:${PYTHONPATH:-}"
export CHALLENGE_TRACK_CODENAME=SENSORS
export DIRECT=1
export UNCERTAINTY_WEIGHT=1
export STOP_CONTROL=1
export TUNED_AIM_DISTANCE=0
export SLOWER=0
export COMPILE=0

REC_PID=""
if [[ "${RECORD_VIDEO}" == "1" ]]; then
  RECORDER_ARGS=(
    python "${ADAPTER_ROOT}/scripts/record_carla_video.py"
    --host 127.0.0.1
    --port "${PORT}"
    --carla-root "${CARLA_ROOT}"
    --view "${VIDEO_VIEW}"
    --output "${VIDEO_OUTPUT}"
    --width "${VIDEO_WIDTH}"
    --height "${VIDEO_HEIGHT}"
    --fps "${VIDEO_FPS}"
    --fov "${VIDEO_FOV}"
    --codec "${VIDEO_CODEC}"
    --sensor-tick "${VIDEO_SENSOR_TICK}"
  )
  if [[ -n "${VIDEO_PIP_VIEW}" ]]; then
    RECORDER_ARGS+=(
      --pip-view "${VIDEO_PIP_VIEW}"
      --pip-scale "${VIDEO_PIP_SCALE}"
      --pip-margin "${VIDEO_PIP_MARGIN}"
      --pip-border "${VIDEO_PIP_BORDER}"
    )
  fi
  "${RECORDER_ARGS[@]}" > "${RECORDER_LOG}" 2>&1 &
  REC_PID=$!
fi

cleanup() {
  if [[ -n "${REC_PID}" ]]; then
    kill -TERM "${REC_PID}" 2>/dev/null || true
    wait "${REC_PID}" 2>/dev/null || true
  fi
  if [[ "${VIDEO_NOTION_COMPAT}" == "1" && -s "${VIDEO_OUTPUT}" ]]; then
    python "${ADAPTER_ROOT}/scripts/convert_video_for_notion.py" \
      --input "${VIDEO_OUTPUT}" \
      --crf "${VIDEO_NOTION_CRF}" \
      --preset "${VIDEO_NOTION_PRESET}" \
      >> "${RECORDER_LOG}" 2>&1 || true
  fi
}
trap cleanup EXIT

cd "${GARAGE_ROOT}"

python "${GARAGE_ROOT}/leaderboard/leaderboard/leaderboard_evaluator.py" \
  --host localhost \
  --port "${PORT}" \
  --traffic-manager-port "${TM_PORT}" \
  --traffic-manager-seed "${SEED}" \
  --routes "${ROUTE_XML}" \
  --repetitions 1 \
  --track SENSORS \
  --agent "${AGENT_PATH}" \
  --agent-config "${TEAM_CONFIG}" \
  --checkpoint "${CHECKPOINT_OUTPUT}" \
  --debug "${DEBUG}" \
  --timeout 900
