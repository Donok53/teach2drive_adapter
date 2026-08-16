#!/usr/bin/env bash
# Frozen TF++ closed-loop baseline with the shifted target sensor rig.
# No adapter, feature hook, controller correction, or offline converted data is used.
set -euo pipefail

ADAPTER_ROOT=${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}
CARLA_ROOT=${CARLA_ROOT:-/home/byeongjae/carla-simulator}
GARAGE_ROOT=${GARAGE_ROOT:-/home/byeongjae/code/carla_garage}
TEAM_CONFIG=${TEAM_CONFIG:-/home/byeongjae/code/checkpoints/transfuserpp/pretrained_models/all_towns}
MISSION_DIR=${MISSION_DIR:-${ADAPTER_ROOT}/runs/leaderboard_tfpp_missions/routes_validation_01}
EGO_VEHICLE_MODEL=${EGO_VEHICLE_MODEL:-vehicle.tesla.model3}
GPU=${GPU:-1}
PORT=${PORT:-2041}
TM_PORT=${TM_PORT:-8041}
RUN_DIR=${RUN_DIR:?Set RUN_DIR to a new evaluation directory}

mkdir -p "${RUN_DIR}"

PATH="/home/byeongjae/miniconda3/envs/mos4d/bin:${PATH}" \
CUDA_VISIBLE_DEVICES="${GPU}" \
ADAPTER_ROOT="${ADAPTER_ROOT}" \
CARLA_ROOT="${CARLA_ROOT}" \
GARAGE_ROOT="${GARAGE_ROOT}" \
TEAM_CONFIG="${TEAM_CONFIG}" \
MISSION_DIR="${MISSION_DIR}" \
RUN_DIR="${RUN_DIR}" \
START_INDEX=0 \
LIMIT=20 \
MAX_RESTARTS=30 \
FRESH_CARLA_ON_START=1 \
CARLA_WAIT_SEC=150 \
CARLA_QUALITY=Low \
CARLA_EXTRA_ARGS="-graphicsadapter=${GPU} -stdout -FullStdOutLogOutput" \
PORT="${PORT}" \
TM_PORT="${TM_PORT}" \
STOP_ON_TIMEOUT=0 \
STOP_ON_INVALID=0 \
INVALID_RETRY_LIMIT=2 \
DEBUG=1 \
RECORD_VIDEO=0 \
TFPP_AGENT_RECORD_VIDEO=0 \
CLEANUP_AFTER_MISSION=1 \
EGO_VEHICLE_MODEL="${EGO_VEHICLE_MODEL}" \
AGENT_PATH="${ADAPTER_ROOT}/scripts/tfpp_sensor_rig_agent.py" \
TFPP_SENSOR_RIG=front_triplet_shifted \
TFPP_SENSOR_CAMERA=front \
TFPP_SENSOR_LIDAR=top \
TFPP_STEER_CORRECTION=1.0 \
TFPP_ADAPTER_CHECKPOINT= \
TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT= \
MISSION_TIMEOUT_SEC=500 \
bash "${ADAPTER_ROOT}/scripts/run_tfpp_mission_batch_autorestart.sh"
