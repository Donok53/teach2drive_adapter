#!/usr/bin/env bash
set -euo pipefail

HOST_WORK_ROOT=${HOST_WORK_ROOT:-/media/aimlab/HDD00/users/byeongjae/workspace}
HOST_T2D_ROOT=${HOST_T2D_ROOT:-${HOST_WORK_ROOT}/teach2drive}
HOST_ADAPTER_ROOT=${HOST_ADAPTER_ROOT:-${HOST_T2D_ROOT}/code/teach2drive_adapter}
HOST_GARAGE_ROOT=${HOST_GARAGE_ROOT:-${HOST_T2D_ROOT}/code/carla_garage}
HOST_DATA_ROOT=${HOST_DATA_ROOT:-${HOST_T2D_ROOT}/data}
HOST_CARLA_ROOT=${HOST_CARLA_ROOT:-${HOST_WORK_ROOT}/carla/carla_0.9.15}
IMAGE=${IMAGE:-teach2drive-eval-py310:dl2}
GPU=${GPU:-1}
PORT=${PORT:-2053}
TM_PORT=${TM_PORT:-8053}
MISSIONS=${MISSIONS:-2,4,5,8,11,15,17}
BLENDS=${BLENDS:-0.25,0.5,0.0}
RUN_NAME=${RUN_NAME:-eval_vehicle_dynamics_v3_selected_20260814}
HOST_RUN_ROOT=${HOST_RUN_ROOT:-${HOST_DATA_ROOT}/runs/${RUN_NAME}}
HOST_HF_CACHE=${HOST_HF_CACHE:-${HOST_DATA_ROOT}/runtime/huggingface}
CONTAINER_ADAPTER_ROOT=/data/users/byeongjae/code/teach2drive_adapter
CONTAINER_GARAGE_ROOT=/data/users/byeongjae/code/carla_garage
CONTAINER_DATA_ROOT=/data/dataset/byeongjae
CONTAINER_CARLA_ROOT=/carla
CHECKPOINT=${CHECKPOINT:-${CONTAINER_DATA_ROOT}/runs/train_exact_vehicle_dynamics_v3_gpu0/train/best_model.pt}
TEAM_CONFIG=${TEAM_CONFIG:-${CONTAINER_DATA_ROOT}/checkpoints/transfuserpp/pretrained_models/all_towns}
MISSION_DIR=${CONTAINER_ADAPTER_ROOT}/runs/leaderboard_tfpp_missions/routes_validation_01
MISSION_LIST=${CONTAINER_DATA_ROOT}/runs/${RUN_NAME}/mission_routes_container.txt

mkdir -p "${HOST_RUN_ROOT}" "${HOST_HF_CACHE}"
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

start_carla() {
  local log=$1
  stop_carla
  if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    echo "Port ${PORT} is already occupied; refusing to kill an unrelated process." >&2
    return 1
  fi
  (
    cd "${HOST_CARLA_ROOT}"
    exec setsid env DISPLAY=:0 CUDA_VISIBLE_DEVICES="${GPU}" ./CarlaUE4.sh \
      -RenderOffScreen -nosound -quality-level=Low \
      -carla-rpc-port="${PORT}" -graphicsadapter="${GPU}"
  ) > "${log}" 2>&1 &
  SERVER_PID=$!

  for _ in $(seq 1 60); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "CARLA exited during startup. See ${log}" >&2
      return 1
    fi
    if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
      sleep 12
      return 0
    fi
    sleep 1
  done
  echo "CARLA did not open port ${PORT}. See ${log}" >&2
  return 1
}

IFS=',' read -r -a mission_array <<< "${MISSIONS}"
IFS=',' read -r -a blend_array <<< "${BLENDS}"

for mission in "${mission_array[@]}"; do
  for blend in "${blend_array[@]}"; do
    blend_tag=$(python3 -c 'import sys; print(f"b{round(float(sys.argv[1])*100):03d}")' "${blend}")
    host_run_dir="${HOST_RUN_ROOT}/${blend_tag}"
    container_run_dir="${CONTAINER_DATA_ROOT}/runs/${RUN_NAME}/${blend_tag}"
    mkdir -p "${host_run_dir}/carla_logs" "${host_run_dir}/traces"
    carla_log="${host_run_dir}/carla_logs/mission_$(printf '%03d' "${mission}").log"
    echo "[$(date --iso-8601=seconds)] mission=${mission} blend=${blend} starting"
    start_carla "${carla_log}"

    set +e
    docker run --rm --network host --gpus "device=${GPU}" \
      -e CUDA_VISIBLE_DEVICES=0 \
      -e ADAPTER_ROOT="${CONTAINER_ADAPTER_ROOT}" \
      -e CARLA_ROOT="${CONTAINER_CARLA_ROOT}" \
      -e GARAGE_ROOT="${CONTAINER_GARAGE_ROOT}" \
      -e TEAM_CONFIG="${TEAM_CONFIG}" \
      -e MISSION_DIR="${MISSION_DIR}" \
      -e MISSION_LIST="${MISSION_LIST}" \
      -e RUN_DIR="${container_run_dir}" \
      -e START_INDEX="${mission}" \
      -e LIMIT=1 \
      -e PORT="${PORT}" \
      -e TM_PORT="${TM_PORT}" \
      -e MISSION_TIMEOUT_SEC=500 \
      -e STOP_ON_TIMEOUT=0 \
      -e STOP_ON_INVALID=0 \
      -e INVALID_RETRY_LIMIT=1 \
      -e CARLA_WATCHDOG=1 \
      -e CLEANUP_AFTER_MISSION=1 \
      -e RECORD_VIDEO=1 \
      -e VIDEO_RECORD_INDICES="${MISSIONS}" \
      -e VIDEO_VIEW=topdown \
      -e VIDEO_PIP_VIEW=front \
      -e VIDEO_NOTION_COMPAT=1 \
      -e EGO_VEHICLE_MODEL=vehicle.tesla.model3 \
      -e AGENT_PATH="${CONTAINER_ADAPTER_ROOT}/scripts/tfpp_vehicle_dynamics_v3_sensor_rig_agent.py" \
      -e TFPP_SENSOR_RIG=tfpp_ego \
      -e TFPP_SENSOR_CAMERA=front \
      -e TFPP_SENSOR_LIDAR=top \
      -e TFPP_VEHICLE_DYNAMICS_V3_CHECKPOINT="${CHECKPOINT}" \
      -e TFPP_DYNAMICS_TRACE_PATH="${container_run_dir}/traces/dynamics_v3.jsonl" \
      -e TFPP_DYNAMICS_BLEND="${blend}" \
      -e TFPP_DYNAMICS_MAX_DELTA=0.12 \
      -e TFPP_DYNAMICS_TURN_THRESHOLD_YAW=0.03 \
      -e TFPP_DYNAMICS_FULL_TURN_THRESHOLD_YAW=0.12 \
      -e TFPP_DYNAMICS_CURVATURE_PROBE=0.75 \
      -e TFPP_DYNAMICS_RISK_GATE_FLOOR=0.25 \
      -e TFPP_DYNAMICS_MINIMUM_TARGET_SPEED=1.0 \
      -e TFPP_DYNAMICS_LAG1_FRAMES=5 \
      -e TFPP_DYNAMICS_LAG2_FRAMES=10 \
      -e HF_HOME=/cache/huggingface \
      -e PYTHONPATH="/opt/carla_py310:${CONTAINER_ADAPTER_ROOT}:${CONTAINER_GARAGE_ROOT}/team_code" \
      -v "${HOST_T2D_ROOT}/code:/data/users/byeongjae/code" \
      -v "${HOST_DATA_ROOT}:${CONTAINER_DATA_ROOT}" \
      -v "${HOST_CARLA_ROOT}:${CONTAINER_CARLA_ROOT}:ro" \
      -v "${HOST_DATA_ROOT}/runtime/carla_py310:/opt/carla_py310:ro" \
      -v "${HOST_HF_CACHE}:/cache/huggingface" \
      "${IMAGE}" \
      bash "${CONTAINER_ADAPTER_ROOT}/scripts/run_tfpp_mission_batch.sh"
    eval_exit=$?
    set -e
    stop_carla
    echo "[$(date --iso-8601=seconds)] mission=${mission} blend=${blend} exit=${eval_exit}"
  done
done

echo "[$(date --iso-8601=seconds)] selected v3 evaluation complete"
