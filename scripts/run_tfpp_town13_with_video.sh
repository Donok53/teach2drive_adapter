#!/usr/bin/env bash

CARLA_ROOT="${CARLA_ROOT:-/home/byeongjae/carla-simulator}"
GARAGE_ROOT="${GARAGE_ROOT:-/home/byeongjae/code/carla_garage}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}"
TEAM_CONFIG="${TEAM_CONFIG:-/home/byeongjae/code/checkpoints/transfuserpp/pretrained_models/all_towns}"
ROUTE_XML="${ROUTE_XML:-${GARAGE_ROOT}/leaderboard/data/routes_validation_split/routes_validation_00.xml}"
RUN_DIR="${RUN_DIR:-${ADAPTER_ROOT}/runs/leaderboard_tfpp_town13_validation}"
PORT="${PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
SEED="${SEED:-0}"
DEBUG="${DEBUG:-1}"
VIDEO_VIEW="${VIDEO_VIEW:-chase}"
VIDEO_OUTPUT="${VIDEO_OUTPUT:-${RUN_DIR}/routes_validation_00_${VIDEO_VIEW}.mp4}"
CHECKPOINT_OUTPUT="${CHECKPOINT_OUTPUT:-${RUN_DIR}/result_routes_validation_00.json}"

mkdir -p "${RUN_DIR}"

export PATH="/home/byeongjae/miniconda3/envs/mos4d/bin:${PATH}"
export CARLA_ROOT
export WORK_DIR="${GARAGE_ROOT}"
export SCENARIO_RUNNER_ROOT="${GARAGE_ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${GARAGE_ROOT}/leaderboard"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${GARAGE_ROOT}/team_code:${PYTHONPATH:-}"
export CHALLENGE_TRACK_CODENAME=SENSORS
export DIRECT=1
export UNCERTAINTY_WEIGHT=1
export STOP_CONTROL=1
export TUNED_AIM_DISTANCE=0
export SLOWER=0
export COMPILE=0

python "${ADAPTER_ROOT}/scripts/record_carla_video.py" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --carla-root "${CARLA_ROOT}" \
  --view "${VIDEO_VIEW}" \
  --output "${VIDEO_OUTPUT}" \
  > "${RUN_DIR}/recorder.log" 2>&1 &

REC_PID=$!

cleanup() {
  kill -TERM "${REC_PID}" 2>/dev/null || true
  wait "${REC_PID}" 2>/dev/null || true
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
  --agent "${GARAGE_ROOT}/team_code/sensor_agent.py" \
  --agent-config "${TEAM_CONFIG}" \
  --checkpoint "${CHECKPOINT_OUTPUT}" \
  --debug "${DEBUG}" \
  --timeout 900
