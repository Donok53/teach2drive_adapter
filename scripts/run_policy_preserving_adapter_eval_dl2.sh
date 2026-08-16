#!/usr/bin/env bash
set -euo pipefail

HOST_WORK_ROOT=${HOST_WORK_ROOT:-/media/aimlab/HDD00/users/byeongjae/workspace}
HOST_T2D_ROOT=${HOST_T2D_ROOT:-${HOST_WORK_ROOT}/teach2drive}
HOST_ADAPTER_ROOT=${HOST_ADAPTER_ROOT:-${HOST_T2D_ROOT}/code/teach2drive_adapter}
HOST_DATA_ROOT=${HOST_DATA_ROOT:-${HOST_T2D_ROOT}/data}
HOST_CARLA_ROOT=${HOST_CARLA_ROOT:-${HOST_WORK_ROOT}/carla/carla_0.9.15}
IMAGE=${IMAGE:-teach2drive-eval-py310:dl2}
CHECKPOINT=${CHECKPOINT:?Set container-visible CHECKPOINT path}
GPU=${GPU:-0}
PORT=${PORT:-2063}
TM_PORT=${TM_PORT:-8063}
START_INDEX=${START_INDEX:-0}
LIMIT=${LIMIT:-20}
RUN_NAME=${RUN_NAME:-eval_policy_preserving_adapter}
VIDEO_RECORD_INDICES=${VIDEO_RECORD_INDICES:-0,1,2,3,4}
OUTPUT_RESIDUAL_BLEND=${OUTPUT_RESIDUAL_BLEND:-1.0}
HOST_RUN_ROOT=${HOST_DATA_ROOT}/runs/${RUN_NAME}
HOST_HF_CACHE=${HOST_DATA_ROOT}/runtime/huggingface

CONTAINER_ADAPTER_ROOT=/data/users/byeongjae/code/teach2drive_adapter
CONTAINER_GARAGE_ROOT=/data/users/byeongjae/code/carla_garage
CONTAINER_DATA_ROOT=/data/dataset/byeongjae
CONTAINER_CARLA_ROOT=/carla
TEAM_CONFIG=${TEAM_CONFIG:-${CONTAINER_DATA_ROOT}/checkpoints/transfuserpp/pretrained_models/all_towns}
MISSION_DIR=${CONTAINER_ADAPTER_ROOT}/runs/leaderboard_tfpp_missions/routes_validation_01
MISSION_LIST=${CONTAINER_DATA_ROOT}/runs/${RUN_NAME}/mission_routes_container.txt
CONTAINER_RUN_DIR=${CONTAINER_DATA_ROOT}/runs/${RUN_NAME}

mkdir -p "${HOST_RUN_ROOT}/carla_logs" "${HOST_HF_CACHE}"
: > "${HOST_RUN_ROOT}/mission_routes_container.txt"
for route in "${HOST_ADAPTER_ROOT}"/runs/leaderboard_tfpp_missions/routes_validation_01/mission_*.xml; do
  printf '%s/%s\n' "${MISSION_DIR}" "$(basename "${route}")" >> "${HOST_RUN_ROOT}/mission_routes_container.txt"
done

SERVER_PID=""
stop_carla() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill -TERM -- -"${SERVER_PID}" 2>/dev/null || true
    sleep 3
    kill -KILL -- -"${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_carla EXIT INT TERM

if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
  echo "Port ${PORT} is already occupied; refusing to kill an unrelated process." >&2
  exit 1
fi
(
  cd "${HOST_CARLA_ROOT}"
  # RenderOffScreen does not need an X display.  Leaving DISPLAY pointed at
  # :0 makes unattended SSH launches fail when no matching Xauthority exists.
  exec setsid env DISPLAY= CUDA_VISIBLE_DEVICES="${GPU}" ./CarlaUE4.sh \
    -RenderOffScreen -nosound -quality-level=Low \
    -carla-rpc-port="${PORT}" -graphicsadapter="${GPU}"
) > "${HOST_RUN_ROOT}/carla_logs/carla.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 75); do
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "CARLA exited during startup" >&2; exit 1; }
  if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then sleep 12; break; fi
  sleep 1
done

echo "[$(date --iso-8601=seconds)] policy-preserving adapter evaluation starting GPU=${GPU} checkpoint=${CHECKPOINT}"
docker run --rm --network host --gpus "device=${GPU}" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e ADAPTER_ROOT="${CONTAINER_ADAPTER_ROOT}" -e CARLA_ROOT="${CONTAINER_CARLA_ROOT}" \
  -e GARAGE_ROOT="${CONTAINER_GARAGE_ROOT}" -e TEAM_CONFIG="${TEAM_CONFIG}" \
  -e MISSION_DIR="${MISSION_DIR}" -e MISSION_LIST="${MISSION_LIST}" \
  -e RUN_DIR="${CONTAINER_RUN_DIR}" -e START_INDEX="${START_INDEX}" -e LIMIT="${LIMIT}" \
  -e PORT="${PORT}" -e TM_PORT="${TM_PORT}" \
  -e MISSION_TIMEOUT_SEC=500 -e STOP_ON_TIMEOUT=0 -e STOP_ON_INVALID=0 \
  -e INVALID_RETRY_LIMIT=2 -e CARLA_WATCHDOG=1 -e CLEANUP_AFTER_MISSION=1 \
  -e RECORD_VIDEO=1 -e VIDEO_RECORD_INDICES="${VIDEO_RECORD_INDICES}" \
  -e VIDEO_VIEW=topdown -e VIDEO_PIP_VIEW=front -e VIDEO_NOTION_COMPAT=1 \
  -e EGO_VEHICLE_MODEL=vehicle.tesla.model3 \
  -e AGENT_PATH="${CONTAINER_ADAPTER_ROOT}/scripts/tfpp_feature_then_fusion_peft_adapter_sensor_rig_agent.py" \
  -e TFPP_SENSOR_RIG=tfpp_ego -e TFPP_SENSOR_CAMERA=front -e TFPP_SENSOR_LIDAR=top \
  -e TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT="${CHECKPOINT}" \
  -e TFPP_OUTPUT_RESIDUAL_BLEND="${OUTPUT_RESIDUAL_BLEND}" \
  -e TFPP_FEATURE_ADAPTER_BLEND=1.0 -e TFPP_STAGE_FEATURE_ADAPTER_BLEND=1.0 -e TFPP_FUSION_ADAPTER_BLEND=1.0 \
  -e HF_HOME=/cache/huggingface \
  -e PYTHONPATH="/opt/carla_py310:${CONTAINER_ADAPTER_ROOT}:${CONTAINER_GARAGE_ROOT}/team_code" \
  -v "${HOST_T2D_ROOT}/code:/data/users/byeongjae/code" \
  -v "${HOST_DATA_ROOT}:${CONTAINER_DATA_ROOT}" \
  -v "${HOST_CARLA_ROOT}:${CONTAINER_CARLA_ROOT}:ro" \
  -v "${HOST_DATA_ROOT}/runtime/carla_py310:/opt/carla_py310:ro" \
  -v "${HOST_HF_CACHE}:/cache/huggingface" \
  "${IMAGE}" bash "${CONTAINER_ADAPTER_ROOT}/scripts/run_tfpp_mission_batch.sh"

echo "[$(date --iso-8601=seconds)] policy-preserving adapter evaluation complete"
