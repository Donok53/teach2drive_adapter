#!/usr/bin/env bash

set -euo pipefail

export HOME="${HOME:-/home/jovyan}"

ADAPTER_ROOT="${ADAPTER_ROOT:-${HOME}/teach2drive/workspace/teach2drive_adapter}"
CARLA_ROOT="${CARLA_ROOT:-${HOME}/dataset/byeongjae/carla-simulator}"
GARAGE_ROOT="${GARAGE_ROOT:-${HOME}/teach2drive/workspace/carla_garage}"
TEAM_CONFIG="${TEAM_CONFIG:-${HOME}/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns}"
MISSION_DIR="${MISSION_DIR:-${ADAPTER_ROOT}/runs/leaderboard_tfpp_missions/routes_validation_01}"
RUN_DIR="${RUN_DIR:-${HOME}/dataset/byeongjae/runs/eval_task_feature_sgpu_mission1_t180}"
LOG_DIR="${LOG_DIR:-${HOME}/teach2drive/logs}"

CHECKPOINT="${CHECKPOINT:-${HOME}/dataset/byeongjae/runs/tfpp_tesla_town13_task_feature_adapter_sgpu/train_front_triplet_shifted_task_feature_adapter_sgpu/best_model.pt}"
AGENT="${AGENT:-${ADAPTER_ROOT}/scripts/tfpp_feature_then_fusion_peft_adapter_sensor_rig_agent.py}"

START_INDEX="${START_INDEX:-1}"
LIMIT="${LIMIT:-1}"
PORT="${PORT:-2011}"
TM_PORT="${TM_PORT:-8011}"
CARLA_GRAPHICS_ADAPTER="${CARLA_GRAPHICS_ADAPTER:-7}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

if [[ ! -f "${MISSION_DIR}/mission_routes.txt" ]]; then
  echo "missing mission list: ${MISSION_DIR}/mission_routes.txt" >&2
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

cd "${ADAPTER_ROOT}"

ADAPTER_ROOT="${ADAPTER_ROOT}" \
CARLA_ROOT="${CARLA_ROOT}" \
GARAGE_ROOT="${GARAGE_ROOT}" \
TEAM_CONFIG="${TEAM_CONFIG}" \
MISSION_DIR="${MISSION_DIR}" \
RUN_DIR="${RUN_DIR}" \
START_INDEX="${START_INDEX}" \
LIMIT="${LIMIT}" \
MAX_RESTARTS="${MAX_RESTARTS:-2}" \
FRESH_CARLA_ON_START=1 \
CARLA_QUALITY=Low \
CARLA_EXTRA_ARGS="-graphicsadapter=${CARLA_GRAPHICS_ADAPTER} -stdout -FullStdOutLogOutput" \
DEBUG=1 \
RECORD_VIDEO=0 \
TFPP_AGENT_RECORD_VIDEO=1 \
TFPP_RECORD_SENSOR_ID=rgb_front \
TFPP_RECORD_FPS=20 \
TFPP_RECORD_CODEC=mp4v \
TFPP_RECORD_EVERY_N=1 \
VIDEO_EXT=mp4 \
VIDEO_VIEW=input_hq \
PORT="${PORT}" \
TM_PORT="${TM_PORT}" \
STOP_ON_TIMEOUT=0 \
STOP_ON_INVALID=0 \
CLEANUP_AFTER_MISSION=1 \
AGENT_PATH="${AGENT}" \
TFPP_SENSOR_RIG=front_triplet_shifted \
TFPP_SENSOR_CAMERA=front \
TFPP_SENSOR_LIDAR=top \
TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT="${CHECKPOINT}" \
TFPP_FEATURE_ADAPTER_BLEND=1.0 \
TFPP_STAGE_FEATURE_ADAPTER_BLEND=1.0 \
TFPP_FUSION_ADAPTER_BLEND=1.0 \
MISSION_TIMEOUT_SEC="${MISSION_TIMEOUT_SEC:-180}" \
bash scripts/run_tfpp_mission_batch_autorestart.sh
