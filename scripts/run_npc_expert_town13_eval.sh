#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# CARLA Leaderboard-style sanity check with the built-in NPC/BasicAgent route
# follower. This is not a model baseline; it checks whether the local mission
# set is passable by a route-aware CARLA navigation agent.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export EGO_VEHICLE_MODEL=${EGO_VEHICLE_MODEL:-vehicle.tesla.model3}

export ADAPTER_ROOT=${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}
export CARLA_ROOT=${CARLA_ROOT:-/home/byeongjae/carla-simulator}
export GARAGE_ROOT=${GARAGE_ROOT:-/home/byeongjae/code/carla_garage}
export MISSION_DIR=${MISSION_DIR:-"$ADAPTER_ROOT/runs/leaderboard_tfpp_missions/routes_validation_01"}
export RUN_DIR=${RUN_DIR:-"$MISSION_DIR/tfpp_runs_npc_expert_t180_overhead"}
export TEAM_CONFIG=${TEAM_CONFIG:-/home/byeongjae/code/checkpoints/transfuserpp/pretrained_models/all_towns}
export AGENT_PATH=${AGENT_PATH:-"$GARAGE_ROOT/leaderboard/leaderboard/autoagents/npc_agent.py"}

export START_INDEX=${START_INDEX:-0}
export LIMIT=${LIMIT:-20}
export MISSION_TIMEOUT_SEC=${MISSION_TIMEOUT_SEC:-180}
export MAX_RESTARTS=${MAX_RESTARTS:-12}
export FRESH_CARLA_ON_START=${FRESH_CARLA_ON_START:-1}
export CARLA_QUALITY=${CARLA_QUALITY:-Low}
export CARLA_EXTRA_ARGS=${CARLA_EXTRA_ARGS:-"-graphicsadapter=1"}
export DEBUG=${DEBUG:-1}

export RECORD_VIDEO=${RECORD_VIDEO:-1}
export VIDEO_VIEW=${VIDEO_VIEW:-overhead}
export VIDEO_EXT=${VIDEO_EXT:-mp4}
export VIDEO_CODEC=${VIDEO_CODEC:-mp4v}
export VIDEO_WIDTH=${VIDEO_WIDTH:-1280}
export VIDEO_HEIGHT=${VIDEO_HEIGHT:-720}
export VIDEO_FPS=${VIDEO_FPS:-20}

export STOP_ON_TIMEOUT=${STOP_ON_TIMEOUT:-0}
export STOP_ON_INVALID=${STOP_ON_INVALID:-0}
export CLEANUP_AFTER_MISSION=${CLEANUP_AFTER_MISSION:-1}

exec bash scripts/run_tfpp_mission_batch_autorestart.sh
