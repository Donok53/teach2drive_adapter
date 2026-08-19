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
# DL2's Unreal/Vulkan enumeration is reversed relative to nvidia-smi: adapter
# 0 is physical NVIDIA GPU1, which is reserved for evaluation.
CARLA_GRAPHICS_ADAPTER=${CARLA_GRAPHICS_ADAPTER:-0}
PORT=${PORT:-2063}
TM_PORT=${TM_PORT:-8063}
SEED=${SEED:-0}
START_INDEX=${START_INDEX:-0}
LIMIT=${LIMIT:-20}
RUN_NAME=${RUN_NAME:-eval_policy_preserving_adapter}
RECORD_VIDEO=${RECORD_VIDEO:-1}
VIDEO_RECORD_INDICES=${VIDEO_RECORD_INDICES:-0,1,2,3,4}
DEBUG=${DEBUG:-1}
MISSION_TIMEOUT_SEC=${MISSION_TIMEOUT_SEC:-500}
OUTPUT_RESIDUAL_BLEND=${OUTPUT_RESIDUAL_BLEND:-1.0}
OUTPUT_RESIDUAL_HARD_GATE_THRESHOLD=${OUTPUT_RESIDUAL_HARD_GATE_THRESHOLD:-}
OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE_OVERRIDE=${OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE_OVERRIDE:-}
OUTPUT_HAZARD_STOP_GATE_THRESHOLD=${OUTPUT_HAZARD_STOP_GATE_THRESHOLD:-}
OUTPUT_HAZARD_STOP_HOLD_FRAMES=${OUTPUT_HAZARD_STOP_HOLD_FRAMES:-0}
OUTPUT_HAZARD_STOP_MIN_DROP_MPS=${OUTPUT_HAZARD_STOP_MIN_DROP_MPS:-0.5}
OUTPUT_HAZARD_STOP_MIN_EGO_SPEED_MPS=${OUTPUT_HAZARD_STOP_MIN_EGO_SPEED_MPS:-0.0}
OUTPUT_HAZARD_STOP_TARGET_SPEED_MPS=${OUTPUT_HAZARD_STOP_TARGET_SPEED_MPS:-0.0}
LIDAR_SAFETY_ENABLED=${LIDAR_SAFETY_ENABLED:-0}
LIDAR_SAFETY_X_MIN_M=${LIDAR_SAFETY_X_MIN_M:-2.45}
LIDAR_SAFETY_X_MAX_M=${LIDAR_SAFETY_X_MAX_M:-20.0}
LIDAR_SAFETY_Y_HALF_WIDTH_M=${LIDAR_SAFETY_Y_HALF_WIDTH_M:-1.15}
LIDAR_SAFETY_Z_MIN_M=${LIDAR_SAFETY_Z_MIN_M:-0.5}
LIDAR_SAFETY_Z_MAX_M=${LIDAR_SAFETY_Z_MAX_M:-1.8}
LIDAR_SAFETY_MIN_CLUSTER_POINTS=${LIDAR_SAFETY_MIN_CLUSTER_POINTS:-6}
LIDAR_SAFETY_CLUSTER_DEPTH_M=${LIDAR_SAFETY_CLUSTER_DEPTH_M:-0.8}
LIDAR_SAFETY_STANDSTILL_BUFFER_M=${LIDAR_SAFETY_STANDSTILL_BUFFER_M:-1.0}
LIDAR_SAFETY_REACTION_TIME_S=${LIDAR_SAFETY_REACTION_TIME_S:-0.3}
LIDAR_SAFETY_DECEL_MPS2=${LIDAR_SAFETY_DECEL_MPS2:-4.5}
LIDAR_SAFETY_MIN_SPEED_DROP_MPS=${LIDAR_SAFETY_MIN_SPEED_DROP_MPS:-0.5}
LIDAR_SAFETY_GATE_THRESHOLD=${LIDAR_SAFETY_GATE_THRESHOLD:-}
OUTPUT_RESIDUAL_MODEL_INDICES=${OUTPUT_RESIDUAL_MODEL_INDICES:-}
OUTPUT_RESIDUAL_MODEL_NAMES=${OUTPUT_RESIDUAL_MODEL_NAMES:-}
OUTPUT_TRACE=${OUTPUT_TRACE:-0}
MISSION_INDICES=${MISSION_INDICES:-}
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
AGENT_PATH=${AGENT_PATH:-${CONTAINER_ADAPTER_ROOT}/scripts/tfpp_feature_then_fusion_peft_adapter_sensor_rig_agent.py}
EGO_VEHICLE_MODEL=${EGO_VEHICLE_MODEL:-vehicle.tesla.model3}
TFPP_MODEL_ORDER=${TFPP_MODEL_ORDER:-}
TFPP_VEHICLE_DYNAMICS_CHECKPOINT=${TFPP_VEHICLE_DYNAMICS_CHECKPOINT:-}
TFPP_DYNAMICS_BLEND=${TFPP_DYNAMICS_BLEND:-0.5}

mkdir -p "${HOST_RUN_ROOT}/carla_logs" "${HOST_HF_CACHE}"

{
  printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'host=%s\n' "$(hostname)"
  printf 'gpu=%s\ncarla_graphics_adapter=%s\nport=%s\ntm_port=%s\nseed=%s\n' \
    "${GPU}" "${CARLA_GRAPHICS_ADAPTER}" "${PORT}" "${TM_PORT}" "${SEED}"
  printf 'agent_path=%s\ncheckpoint=%s\nego_vehicle_model=%s\n' "${AGENT_PATH}" "${CHECKPOINT}" "${EGO_VEHICLE_MODEL}"
  printf 'tfpp_model_order=%s\n' "${TFPP_MODEL_ORDER}"
  printf 'vehicle_dynamics_checkpoint=%s\ndynamics_blend=%s\n' "${TFPP_VEHICLE_DYNAMICS_CHECKPOINT}" "${TFPP_DYNAMICS_BLEND}"
  printf 'mission_indices=%s\nstart_index=%s\nlimit=%s\n' "${MISSION_INDICES}" "${START_INDEX}" "${LIMIT}"
  printf 'debug=%s\nmission_timeout_sec=%s\nrecord_video=%s\n' "${DEBUG}" "${MISSION_TIMEOUT_SEC}" "${RECORD_VIDEO}"
  printf 'output_residual_blend=%s\nhard_gate_threshold=%s\n' "${OUTPUT_RESIDUAL_BLEND}" "${OUTPUT_RESIDUAL_HARD_GATE_THRESHOLD}"
  printf 'speed_logit_scale_override=%s\n' "${OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE_OVERRIDE}"
  printf 'hazard_stop_gate_threshold=%s\nhazard_stop_hold_frames=%s\nhazard_stop_min_drop_mps=%s\nhazard_stop_min_ego_speed_mps=%s\nhazard_stop_target_speed_mps=%s\n' \
    "${OUTPUT_HAZARD_STOP_GATE_THRESHOLD}" "${OUTPUT_HAZARD_STOP_HOLD_FRAMES}" \
    "${OUTPUT_HAZARD_STOP_MIN_DROP_MPS}" "${OUTPUT_HAZARD_STOP_MIN_EGO_SPEED_MPS}" \
    "${OUTPUT_HAZARD_STOP_TARGET_SPEED_MPS}"
  printf 'lidar_safety=%s x=%s:%s y=%s z=%s:%s cluster=%s/%s buffer=%s reaction=%s decel=%s min_drop=%s gate=%s\n' \
    "${LIDAR_SAFETY_ENABLED}" "${LIDAR_SAFETY_X_MIN_M}" "${LIDAR_SAFETY_X_MAX_M}" \
    "${LIDAR_SAFETY_Y_HALF_WIDTH_M}" "${LIDAR_SAFETY_Z_MIN_M}" "${LIDAR_SAFETY_Z_MAX_M}" \
    "${LIDAR_SAFETY_MIN_CLUSTER_POINTS}" "${LIDAR_SAFETY_CLUSTER_DEPTH_M}" \
    "${LIDAR_SAFETY_STANDSTILL_BUFFER_M}" "${LIDAR_SAFETY_REACTION_TIME_S}" \
    "${LIDAR_SAFETY_DECEL_MPS2}" "${LIDAR_SAFETY_MIN_SPEED_DROP_MPS}" \
    "${LIDAR_SAFETY_GATE_THRESHOLD}"
  printf 'adapter_git_head=%s\n' "$(git -C "${HOST_ADAPTER_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  printf 'adapter_git_diff_hash=%s\n' "$(git -C "${HOST_ADAPTER_ROOT}" diff --binary 2>/dev/null | sha256sum | awk '{print $1}')"
  printf 'garage_git_head=%s\n' "$(git -C "${HOST_T2D_ROOT}/code/carla_garage" rev-parse HEAD 2>/dev/null || true)"
  printf 'garage_git_diff_hash=%s\n' "$(git -C "${HOST_T2D_ROOT}/code/carla_garage" diff --binary 2>/dev/null | sha256sum | awk '{print $1}')"
  printf 'docker_image=%s\n' "$(docker image inspect "${IMAGE}" --format '{{.Id}}' 2>/dev/null || true)"
  printf 'checkpoint_sha256=%s\n' "$(sha256sum "${HOST_DATA_ROOT}${CHECKPOINT#${CONTAINER_DATA_ROOT}}" 2>/dev/null | awk '{print $1}')"
  printf 'vehicle_dynamics_sha256=%s\n' "$(sha256sum "${HOST_DATA_ROOT}${TFPP_VEHICLE_DYNAMICS_CHECKPOINT#${CONTAINER_DATA_ROOT}}" 2>/dev/null | awk '{print $1}')"
  printf '%s\n' 'gpu_inventory_begin'
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv,noheader 2>/dev/null || true
  printf '%s\n' 'gpu_inventory_end'
} > "${HOST_RUN_ROOT}/run_manifest.txt"

: > "${HOST_RUN_ROOT}/mission_routes_container.txt"
route_index=-1
for route in "${HOST_ADAPTER_ROOT}"/runs/leaderboard_tfpp_missions/routes_validation_01/mission_*.xml; do
  route_index=$((route_index + 1))
  if [[ -n "${MISSION_INDICES}" && ",${MISSION_INDICES}," != *",${route_index},"* ]]; then
    continue
  fi
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
  # Unreal's -graphicsadapter uses the physical Vulkan adapter index.  Hiding
  # devices with CUDA_VISIBLE_DEVICES at the same time re-indexes/falls back to
  # GPU0 on this host, so leave the physical inventory visible here.  The
  # evaluator container below is still isolated with --gpus device=${GPU}.
  exec setsid env DISPLAY= ./CarlaUE4.sh \
    -RenderOffScreen -nosound -quality-level=Low \
    -carla-rpc-port="${PORT}" -graphicsadapter="${CARLA_GRAPHICS_ADAPTER}"
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
  -e PORT="${PORT}" -e TM_PORT="${TM_PORT}" -e SEED="${SEED}" \
  -e DEBUG="${DEBUG}" -e MISSION_TIMEOUT_SEC="${MISSION_TIMEOUT_SEC}" \
  -e STOP_ON_TIMEOUT=0 -e STOP_ON_INVALID=0 \
  -e INVALID_RETRY_LIMIT=2 -e CARLA_WATCHDOG=1 -e CLEANUP_AFTER_MISSION=1 \
  -e RECORD_VIDEO="${RECORD_VIDEO}" -e VIDEO_RECORD_INDICES="${VIDEO_RECORD_INDICES}" \
  -e VIDEO_VIEW=topdown -e VIDEO_PIP_VIEW=front -e VIDEO_NOTION_COMPAT=1 \
  -e EGO_VEHICLE_MODEL="${EGO_VEHICLE_MODEL}" \
  -e TFPP_MODEL_ORDER="${TFPP_MODEL_ORDER}" \
  -e TFPP_VEHICLE_DYNAMICS_CHECKPOINT="${TFPP_VEHICLE_DYNAMICS_CHECKPOINT}" \
  -e TFPP_DYNAMICS_BLEND="${TFPP_DYNAMICS_BLEND}" \
  -e TFPP_DYNAMICS_TRACE_PATH="${CONTAINER_RUN_DIR}/traces/dynamics.jsonl" \
  -e AGENT_PATH="${AGENT_PATH}" \
  -e TFPP_SENSOR_RIG=tfpp_ego -e TFPP_SENSOR_CAMERA=front -e TFPP_SENSOR_LIDAR=top \
  -e TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT="${CHECKPOINT}" \
  -e TFPP_OUTPUT_RESIDUAL_BLEND="${OUTPUT_RESIDUAL_BLEND}" \
  -e TFPP_OUTPUT_RESIDUAL_HARD_GATE_THRESHOLD="${OUTPUT_RESIDUAL_HARD_GATE_THRESHOLD}" \
  -e TFPP_OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE_OVERRIDE="${OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE_OVERRIDE}" \
  -e TFPP_OUTPUT_HAZARD_STOP_GATE_THRESHOLD="${OUTPUT_HAZARD_STOP_GATE_THRESHOLD}" \
  -e TFPP_OUTPUT_HAZARD_STOP_HOLD_FRAMES="${OUTPUT_HAZARD_STOP_HOLD_FRAMES}" \
  -e TFPP_OUTPUT_HAZARD_STOP_MIN_DROP_MPS="${OUTPUT_HAZARD_STOP_MIN_DROP_MPS}" \
  -e TFPP_OUTPUT_HAZARD_STOP_MIN_EGO_SPEED_MPS="${OUTPUT_HAZARD_STOP_MIN_EGO_SPEED_MPS}" \
  -e TFPP_OUTPUT_HAZARD_STOP_TARGET_SPEED_MPS="${OUTPUT_HAZARD_STOP_TARGET_SPEED_MPS}" \
  -e TFPP_LIDAR_SAFETY_ENABLED="${LIDAR_SAFETY_ENABLED}" \
  -e TFPP_LIDAR_SAFETY_X_MIN_M="${LIDAR_SAFETY_X_MIN_M}" \
  -e TFPP_LIDAR_SAFETY_X_MAX_M="${LIDAR_SAFETY_X_MAX_M}" \
  -e TFPP_LIDAR_SAFETY_Y_HALF_WIDTH_M="${LIDAR_SAFETY_Y_HALF_WIDTH_M}" \
  -e TFPP_LIDAR_SAFETY_Z_MIN_M="${LIDAR_SAFETY_Z_MIN_M}" \
  -e TFPP_LIDAR_SAFETY_Z_MAX_M="${LIDAR_SAFETY_Z_MAX_M}" \
  -e TFPP_LIDAR_SAFETY_MIN_CLUSTER_POINTS="${LIDAR_SAFETY_MIN_CLUSTER_POINTS}" \
  -e TFPP_LIDAR_SAFETY_CLUSTER_DEPTH_M="${LIDAR_SAFETY_CLUSTER_DEPTH_M}" \
  -e TFPP_LIDAR_SAFETY_STANDSTILL_BUFFER_M="${LIDAR_SAFETY_STANDSTILL_BUFFER_M}" \
  -e TFPP_LIDAR_SAFETY_REACTION_TIME_S="${LIDAR_SAFETY_REACTION_TIME_S}" \
  -e TFPP_LIDAR_SAFETY_DECEL_MPS2="${LIDAR_SAFETY_DECEL_MPS2}" \
  -e TFPP_LIDAR_SAFETY_MIN_SPEED_DROP_MPS="${LIDAR_SAFETY_MIN_SPEED_DROP_MPS}" \
  -e TFPP_LIDAR_SAFETY_GATE_THRESHOLD="${LIDAR_SAFETY_GATE_THRESHOLD}" \
  -e TFPP_OUTPUT_RESIDUAL_MODEL_INDICES="${OUTPUT_RESIDUAL_MODEL_INDICES}" \
  -e TFPP_OUTPUT_RESIDUAL_MODEL_NAMES="${OUTPUT_RESIDUAL_MODEL_NAMES}" \
  -e TFPP_OUTPUT_TRACE="${OUTPUT_TRACE}" \
  -e TFPP_PRED_BOX_TRACE_PATH="${CONTAINER_RUN_DIR}/traces/pred_boxes.jsonl" \
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
