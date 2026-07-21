#!/usr/bin/env bash
set -euo pipefail

# CARLA server must already be running on HOST:PORT.
# Full PDM-Lite route pool collection, keeping PDM-Lite data format and frame
# cadence while changing only ego vehicle and sensor pose:
#   - ego vehicle: vehicle.tesla.model3
#   - sensor rig source: front_triplet_shifted
#   - PDM-Lite frame/schema output via DataAgentSensorRig

cd "$(dirname "$0")/.."
ADAPTER_ROOT=$(pwd)

export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-2000}
export TM_PORT=${TM_PORT:-8000}
export TM_SEED=${TM_SEED:-42}
export PY=${PY:-python3}

export CARLA_ROOT=${CARLA_ROOT:-"$HOME/carla-simulator"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/code/carla_garage"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/datasets/pdm_lite_tesla_front_triplet_shifted_full"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/pdm_lite_tesla_front_triplet_shifted_full_collect"}
export DATASET_NAME=${DATASET_NAME:-pdm_lite_tesla_front_triplet_shifted_full}
export EGO_VEHICLE_MODEL=${EGO_VEHICLE_MODEL:-vehicle.tesla.model3}
export TFPP_SENSOR_RIG=${TFPP_SENSOR_RIG:-front_triplet_shifted}
export TFPP_SENSOR_CAMERA=${TFPP_SENSOR_CAMERA:-front}
export TFPP_SENSOR_LIDAR=${TFPP_SENSOR_LIDAR:-top}

export TOWNS_CSV=${TOWNS_CSV:-Town01,Town02,Town03,Town04,Town05,Town10HD,Town12,Town13}
export ROUTE_SEED=${ROUTE_SEED:-42}
export ROUTE_ROOT=${ROUTE_ROOT:-"$GARAGE_ROOT/data"}
export ROUTE_SUBSET_XML=${ROUTE_SUBSET_XML:-"$WORK_ROOT/routes/${DATASET_NAME}_routes.xml"}
export ROUTE_SUBSET_META=${ROUTE_SUBSET_META:-"$WORK_ROOT/routes/${DATASET_NAME}_routes.json"}
export TOTAL_ROUTES=${TOTAL_ROUTES:-999999}
export REFRESH_ROUTE_SUBSET=${REFRESH_ROUTE_SUBSET:-0}
export TIMEOUT=${TIMEOUT:-600}
export RESUME=${RESUME:-1}
export ROUTES_SUBSET=${ROUTES_SUBSET:-}
export TOWN=${TOWN:-pdm_lite_tesla_front_triplet_shifted_full}
export DEBUG_CHALLENGE=${DEBUG_CHALLENGE:-0}
export REPETITION=${REPETITION:-0}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export SAVE_PATH="$OUTPUT_ROOT/data"
export CHECKPOINT_ENDPOINT=${CHECKPOINT_ENDPOINT:-"$OUTPUT_ROOT/results/${DATASET_NAME}_result.json"}
export DEBUG_CHECKPOINT=${DEBUG_CHECKPOINT:-"$OUTPUT_ROOT/results/${DATASET_NAME}_live.txt"}

mkdir -p "$OUTPUT_ROOT/data" "$OUTPUT_ROOT/results" "$WORK_ROOT/routes"

if [[ "$REFRESH_ROUTE_SUBSET" == "1" || ! -s "$ROUTE_SUBSET_XML" ]]; then
  "$PY" "$ADAPTER_ROOT/scripts/build_pdm_lite_route_subset.py" \
    --route-root "$ROUTE_ROOT" \
    --output "$ROUTE_SUBSET_XML" \
    --meta-output "$ROUTE_SUBSET_META" \
    --towns-csv "$TOWNS_CSV" \
    --total-routes "$TOTAL_ROUTES" \
    --seed "$ROUTE_SEED"
fi

selected_total_routes=$("$PY" - "$ROUTE_SUBSET_META" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f)["selected_total_routes"])
PY
)

cat > "$OUTPUT_ROOT/collection_plan.json" <<EOF
{
  "dataset": "$DATASET_NAME",
  "collection_style": "carla_garage_pdm_lite_route_expert",
  "ego_vehicle_model": "$EGO_VEHICLE_MODEL",
  "sensor_rig": "$TFPP_SENSOR_RIG",
  "sensor_camera": "$TFPP_SENSOR_CAMERA",
  "sensor_lidar": "$TFPP_SENSOR_LIDAR",
  "towns": "$TOWNS_CSV",
  "selected_total_routes": $selected_total_routes,
  "route_subset_xml": "$ROUTE_SUBSET_XML",
  "route_subset_meta": "$ROUTE_SUBSET_META",
  "output_root": "$OUTPUT_ROOT",
  "save_path": "$SAVE_PATH",
  "checkpoint": "$CHECKPOINT_ENDPOINT"
}
EOF

cd "$GARAGE_ROOT"

export SCENARIO_RUNNER_ROOT="$GARAGE_ROOT/scenario_runner_autopilot"
export LEADERBOARD_ROOT="$GARAGE_ROOT/leaderboard_autopilot"
export CARLA_SERVER="$CARLA_ROOT/CarlaUE4.sh"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:$CARLA_ROOT/PythonAPI/carla:$GARAGE_ROOT/leaderboard_autopilot:$GARAGE_ROOT/scenario_runner_autopilot:$GARAGE_ROOT/team_code:$ADAPTER_ROOT/scripts:${PYTHONPATH:-}"
export REPETITIONS=1
export TEAM_AGENT="$ADAPTER_ROOT/scripts/pdm_lite_data_agent_sensor_rig.py"
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES="$ROUTE_SUBSET_XML"
export TEAM_CONFIG="$ROUTE_SUBSET_XML"
export DATAGEN=1

echo "=== full PDM-Lite Tesla front_triplet_shifted collection"
echo "GARAGE_ROOT=$GARAGE_ROOT"
echo "CARLA_ROOT=$CARLA_ROOT"
echo "EGO_VEHICLE_MODEL=$EGO_VEHICLE_MODEL"
echo "TFPP_SENSOR_RIG=$TFPP_SENSOR_RIG camera=$TFPP_SENSOR_CAMERA lidar=$TFPP_SENSOR_LIDAR"
echo "ROUTES=$ROUTES"
echo "ROUTES_SUBSET=${ROUTES_SUBSET:-<all>}"
echo "SELECTED_TOTAL_ROUTES=$selected_total_routes"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "CHECKPOINT_ENDPOINT=$CHECKPOINT_ENDPOINT"

route_subset_args=()
if [[ -n "$ROUTES_SUBSET" ]]; then
  route_subset_args=(--routes-subset "$ROUTES_SUBSET")
fi

"$PY" leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py \
  --host "$HOST" \
  --port "$PORT" \
  --traffic-manager-port "$TM_PORT" \
  --traffic-manager-seed "$TM_SEED" \
  --routes "$ROUTE_SUBSET_XML" \
  "${route_subset_args[@]}" \
  --repetitions 1 \
  --track MAP \
  --checkpoint "$CHECKPOINT_ENDPOINT" \
  --debug-checkpoint "$DEBUG_CHECKPOINT" \
  --agent "$TEAM_AGENT" \
  --agent-config "$ROUTE_SUBSET_XML" \
  --debug 0 \
  --resume "$RESUME" \
  --timeout "$TIMEOUT"

frames=$(find "$SAVE_PATH" -path '*/measurements/*.json.gz' | wc -l | tr -d ' ')
echo "=== saved_measurement_frames=$frames"
