#!/usr/bin/env bash
set -euo pipefail

# Closed-loop smoke test for the frozen CARLA Garage TransFuser++ baseline.
# CARLA must already be running with rendering enabled, e.g. -RenderOffScreen.

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-2000}
MAP=${MAP:-Town10HD_Opt}
CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}
TFPP_CHECKPOINT=${TFPP_CHECKPOINT:-}
BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-"$HOME/teach2drive/workspace/teach2drive_bootstrap"}
ROUTE_SOURCE=${ROUTE_SOURCE:-"$BOOTSTRAP_ROOT/runs/town10_3cam_640x360_tokens/token_index.npz"}
EPISODE_INDEX=${EPISODE_INDEX:-0}
START_INDEX=${START_INDEX:-30}
DURATION_SEC=${DURATION_SEC:-180}
HZ=${HZ:-10}
VEHICLE_FILTER=${VEHICLE_FILTER:-vehicle.tesla.model3}
OUTPUT=${OUTPUT:-"$HOME/dataset/byeongjae/runs/tfpp_baseline_smoke/episode_${EPISODE_INDEX}.json"}
VIDEO_OUTPUT=${VIDEO_OUTPUT:-"$HOME/dataset/byeongjae/runs/tfpp_baseline_smoke/episode_${EPISODE_INDEX}.mp4"}
TFPP_SPEED_MODE=${TFPP_SPEED_MODE:-expected}
VIDEO_VIEW=${VIDEO_VIEW:-topdown}

if [[ ! -d "$GARAGE_ROOT/team_code" ]]; then
  echo "Missing CARLA Garage team_code: $GARAGE_ROOT/team_code" >&2
  echo "Set GARAGE_ROOT or run configs/transfuserpp_install_local.sh first." >&2
  exit 2
fi

if [[ ! -d "$TEAM_CONFIG" ]]; then
  echo "Missing TransFuser++ team config: $TEAM_CONFIG" >&2
  echo "Set TEAM_CONFIG to a pretrained model folder containing config.json and model_*.pth." >&2
  exit 2
fi

if [[ ! -e "$ROUTE_SOURCE" ]]; then
  echo "Missing route source: $ROUTE_SOURCE" >&2
  echo "Set ROUTE_SOURCE to a token episode directory or token_index.npz." >&2
  exit 2
fi

export CARLA_ROOT
export TEACH2DRIVE_BOOTSTRAP_ROOT="$BOOTSTRAP_ROOT"

ARGS=(
  --host "$HOST"
  --port "$PORT"
  --map "$MAP"
  --baseline-only
  --route-source "$ROUTE_SOURCE"
  --episode-index "$EPISODE_INDEX"
  --start-index "$START_INDEX"
  --duration-sec "$DURATION_SEC"
  --hz "$HZ"
  --output "$OUTPUT"
  --garage-root "$GARAGE_ROOT"
  --team-config "$TEAM_CONFIG"
  --sensor-preset transfuserpp
  --control-mode tfpp_pid
  --command-mode target_angle
  --tfpp-speed-mode "$TFPP_SPEED_MODE"
  --vehicle-filter "$VEHICLE_FILTER"
  --video-output "$VIDEO_OUTPUT"
  --video-view "$VIDEO_VIEW"
  --video-image-size 640 360
  --report-every-sec 5
)

if [[ -n "$TFPP_CHECKPOINT" ]]; then
  ARGS+=(--tfpp-checkpoint "$TFPP_CHECKPOINT")
fi

python -m teach2drive_adapter.carla_rollout_transfuserpp_cached_adapter "${ARGS[@]}"
