#!/usr/bin/env bash
set -euo pipefail

# Canonical-sensor Stage-1 evaluation for the expert future-path lateral
# planner-output adapter.  Five missions record video; the remaining fifteen
# run without video through the batch runner's VIDEO_RECORD_INDICES filter.

ADAPTER_ROOT=${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}
CARLA_ROOT=${CARLA_ROOT:-/home/byeongjae/carla-simulator}
GARAGE_ROOT=${GARAGE_ROOT:-/home/byeongjae/code/carla_garage}
TEAM_CONFIG=${TEAM_CONFIG:-/home/byeongjae/code/checkpoints/transfuserpp/pretrained_models/all_towns}
MISSION_DIR=${MISSION_DIR:-${ADAPTER_ROOT}/runs/leaderboard_tfpp_missions/routes_validation_01}
RUN_DIR=${RUN_DIR:?Set RUN_DIR to a new evaluation directory}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to a Stage-1 checkpoint}
GPU=${GPU:-1}
PORT=${PORT:-2043}
TM_PORT=${TM_PORT:-8043}
START_INDEX=${START_INDEX:-0}
LIMIT=${LIMIT:-20}
VIDEO_RECORD_INDICES=${VIDEO_RECORD_INDICES:-0,1,2,3,4}
OUTPUT_RESIDUAL_BLEND=${OUTPUT_RESIDUAL_BLEND:-1.0}
MISSION_TIMEOUT_SEC=${MISSION_TIMEOUT_SEC:-500}

mkdir -p "${RUN_DIR}"

echo "=== expert-spatial Stage-1 target20 eval start $(date '+%F %T') ==="
echo "run_dir=${RUN_DIR} checkpoint=${CHECKPOINT}"
echo "gpu=${GPU} start_index=${START_INDEX} limit=${LIMIT} video_indices=${VIDEO_RECORD_INDICES} residual_blend=${OUTPUT_RESIDUAL_BLEND} mission_timeout_sec=${MISSION_TIMEOUT_SEC}"

PATH="/home/byeongjae/miniconda3/envs/mos4d/bin:${PATH}" \
CUDA_VISIBLE_DEVICES="${GPU}" \
ADAPTER_ROOT="${ADAPTER_ROOT}" \
CARLA_ROOT="${CARLA_ROOT}" \
GARAGE_ROOT="${GARAGE_ROOT}" \
TEAM_CONFIG="${TEAM_CONFIG}" \
MISSION_DIR="${MISSION_DIR}" \
RUN_DIR="${RUN_DIR}" \
START_INDEX="${START_INDEX}" \
LIMIT="${LIMIT}" \
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
RECORD_VIDEO=1 \
VIDEO_RECORD_INDICES="${VIDEO_RECORD_INDICES}" \
VIDEO_VIEW=topdown \
VIDEO_PIP_VIEW=front \
VIDEO_PIP_SCALE=0.30 \
VIDEO_NOTION_COMPAT=1 \
VIDEO_CODEC=mp4v \
VIDEO_WIDTH=1280 \
VIDEO_HEIGHT=720 \
VIDEO_FPS=20 \
TFPP_AGENT_RECORD_VIDEO=0 \
CLEANUP_AFTER_MISSION=1 \
EGO_VEHICLE_MODEL=vehicle.tesla.model3 \
AGENT_PATH="${ADAPTER_ROOT}/scripts/tfpp_feature_then_fusion_peft_adapter_sensor_rig_agent.py" \
TFPP_SENSOR_RIG=tfpp_ego \
TFPP_SENSOR_CAMERA=front \
TFPP_SENSOR_LIDAR=top \
TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT="${CHECKPOINT}" \
TFPP_OUTPUT_RESIDUAL_BLEND="${OUTPUT_RESIDUAL_BLEND}" \
TFPP_FEATURE_ADAPTER_BLEND=1.0 \
TFPP_STAGE_FEATURE_ADAPTER_BLEND=1.0 \
TFPP_FUSION_ADAPTER_BLEND=1.0 \
MISSION_TIMEOUT_SEC="${MISSION_TIMEOUT_SEC}" \
bash "${ADAPTER_ROOT}/scripts/run_tfpp_mission_batch_autorestart.sh"
