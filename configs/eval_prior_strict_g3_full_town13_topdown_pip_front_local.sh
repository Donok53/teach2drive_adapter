#!/usr/bin/env bash

set -euo pipefail

export PATH="/home/byeongjae/miniconda3/envs/mos4d/bin:${PATH}"

export ADAPTER_ROOT="${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}"
export CARLA_ROOT="${CARLA_ROOT:-/home/byeongjae/carla-simulator}"
export GARAGE_ROOT="${GARAGE_ROOT:-/home/byeongjae/code/carla_garage}"
export TEAM_CONFIG="${TEAM_CONFIG:-/home/byeongjae/code/checkpoints/transfuserpp/pretrained_models/all_towns}"
export AGENT_PATH="${AGENT_PATH:-${ADAPTER_ROOT}/scripts/tfpp_feature_then_fusion_peft_adapter_sensor_rig_agent.py}"

export MISSION_DIR="${MISSION_DIR:-${ADAPTER_ROOT}/runs/leaderboard_tfpp_missions/routes_validation_01}"
export RUN_DIR="${RUN_DIR:-${ADAPTER_ROOT}/runs/eval_prior_strict_g3_restart_full_town13_topdown_pip_front_local}"
export SUMMARY_TSV="${SUMMARY_TSV:-${RUN_DIR}/summary.tsv}"
export TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT="${TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT:-${ADAPTER_ROOT}/runs/remote_checkpoints/prior_strict_g3_restart/best_model.pt}"

export PORT="${PORT:-2011}"
export TM_PORT="${TM_PORT:-8011}"
export START_INDEX="${START_INDEX:-0}"
export LIMIT="${LIMIT:-0}"
export MAX_RESTARTS="${MAX_RESTARTS:-30}"
export CARLA_WAIT_SEC="${CARLA_WAIT_SEC:-150}"
export CARLA_QUALITY="${CARLA_QUALITY:-Low}"
export CARLA_EXTRA_ARGS="${CARLA_EXTRA_ARGS:--graphicsadapter=1 -stdout -FullStdOutLogOutput}"
export FRESH_CARLA_ON_START="${FRESH_CARLA_ON_START:-1}"
export STOP_ON_INVALID="${STOP_ON_INVALID:-0}"
export STOP_ON_TIMEOUT="${STOP_ON_TIMEOUT:-0}"
export MISSION_TIMEOUT_SEC="${MISSION_TIMEOUT_SEC:-420}"

export EGO_VEHICLE_MODEL="${EGO_VEHICLE_MODEL:-vehicle.tesla.model3}"
export RECORD_VIDEO="${RECORD_VIDEO:-1}"
export VIDEO_VIEW="${VIDEO_VIEW:-topdown}"
export VIDEO_PIP_VIEW="${VIDEO_PIP_VIEW:-front}"
export VIDEO_PIP_SCALE="${VIDEO_PIP_SCALE:-0.30}"
export VIDEO_PIP_MARGIN="${VIDEO_PIP_MARGIN:-24}"
export VIDEO_PIP_BORDER="${VIDEO_PIP_BORDER:-2}"
export VIDEO_WIDTH="${VIDEO_WIDTH:-1280}"
export VIDEO_HEIGHT="${VIDEO_HEIGHT:-720}"
export VIDEO_FPS="${VIDEO_FPS:-20}"
export VIDEO_CODEC="${VIDEO_CODEC:-mp4v}"
export VIDEO_NOTION_COMPAT="${VIDEO_NOTION_COMPAT:-1}"
export VIDEO_NOTION_CRF="${VIDEO_NOTION_CRF:-23}"
export VIDEO_NOTION_PRESET="${VIDEO_NOTION_PRESET:-medium}"

export DEBUG="${LEADERBOARD_DEBUG:-1}"
export TFPP_SENSOR_RIG="${TFPP_SENSOR_RIG:-front_triplet_shifted}"
export TFPP_SENSOR_CAMERA="${TFPP_SENSOR_CAMERA:-front}"
export TFPP_SENSOR_LIDAR="${TFPP_SENSOR_LIDAR:-top}"
export TFPP_FEATURE_ADAPTER_BLEND="${TFPP_FEATURE_ADAPTER_BLEND:-1.0}"
export TFPP_STAGE_FEATURE_ADAPTER_BLEND="${TFPP_STAGE_FEATURE_ADAPTER_BLEND:-1.0}"
export TFPP_FUSION_ADAPTER_BLEND="${TFPP_FUSION_ADAPTER_BLEND:-1.0}"
export TFPP_AGENT_RECORD_VIDEO="${TFPP_AGENT_RECORD_VIDEO:-0}"

mkdir -p "${RUN_DIR}"

echo "=== full Town13 mission eval"
echo "checkpoint=${TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT}"
echo "run_dir=${RUN_DIR}"
echo "video=${VIDEO_VIEW}+pip:${VIDEO_PIP_VIEW} notion=${VIDEO_NOTION_COMPAT}"

bash "${ADAPTER_ROOT}/scripts/run_tfpp_mission_batch_autorestart.sh"
