#!/usr/bin/env bash

set -euo pipefail

cd /home/byeongjae/code/teach2drive_adapter

RUN_DIR=/home/byeongjae/code/teach2drive_adapter/runs/leaderboard_tfpp_missions/routes_validation_01/tfpp_runs_fa_v1_fused_t180_input_video_exact
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-20}"

ADAPTER_ROOT=/home/byeongjae/code/teach2drive_adapter \
CARLA_ROOT=/home/byeongjae/carla-simulator \
MISSION_DIR=/home/byeongjae/code/teach2drive_adapter/runs/leaderboard_tfpp_missions/routes_validation_01 \
RUN_DIR="${RUN_DIR}" \
START_INDEX="${START_INDEX}" \
LIMIT="${LIMIT}" \
MAX_RESTARTS=12 \
FRESH_CARLA_ON_START=1 \
CARLA_QUALITY=Low \
DEBUG=1 \
RECORD_VIDEO=0 \
TFPP_AGENT_RECORD_VIDEO=1 \
TFPP_RECORD_SENSOR_ID=rgb_front \
TFPP_RECORD_FPS=20 \
TFPP_RECORD_CODEC=mp4v \
TFPP_RECORD_EVERY_N=1 \
VIDEO_EXT=mp4 \
VIDEO_VIEW=input_hq \
STOP_ON_TIMEOUT=0 \
STOP_ON_INVALID=0 \
CLEANUP_AFTER_MISSION=1 \
AGENT_PATH=/home/byeongjae/code/teach2drive_adapter/scripts/tfpp_feature_adapter_sensor_rig_agent.py \
TFPP_SENSOR_RIG=front_triplet_shifted \
TFPP_SENSOR_CAMERA=front \
TFPP_SENSOR_LIDAR=top \
TFPP_ADAPTER_CHECKPOINT=/home/byeongjae/code/teach2drive_adapter/runs/tfpp_feature_adapter_v1_fused/train_fused_feature_adapter_v1/best_model.pt \
TFPP_FEATURE_ADAPTER_BLEND=1.0 \
MISSION_TIMEOUT_SEC=180 \
bash scripts/run_tfpp_mission_batch_autorestart.sh
