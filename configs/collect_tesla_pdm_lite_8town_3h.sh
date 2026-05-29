#!/usr/bin/env bash
set -euo pipefail

# CARLA server must already be running on HOST:PORT.
# This collects a small PDM-Lite/TransFuser++-style route dataset:
#   - CARLA Garage PDM-Lite route/scenario XMLs
#   - PDM-Lite data_agent expert
#   - Tesla Model 3 ego vehicle
#   - balanced across the 8 towns used by the local PDM-Lite route pool

cd "$(dirname "$0")/.."
ADAPTER_ROOT=$(pwd)

export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-2000}
export TM_PORT=${TM_PORT:-8000}
export TM_SEED=${TM_SEED:-42}
export PY=${PY:-"$HOME/.venv/carla37/bin/python"}
if [[ ! -x "$PY" ]]; then
  PY=python
fi

export CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/datasets/pdm_lite_tesla_8town_3h"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/pdm_lite_tesla_8town_3h_collect"}
export EGO_VEHICLE_MODEL=${EGO_VEHICLE_MODEL:-vehicle.tesla.model3}

export TARGET_HOURS=${TARGET_HOURS:-3.0}
export AVG_ROUTE_SEC=${AVG_ROUTE_SEC:-28.0}
export TOWNS_CSV=${TOWNS_CSV:-Town01,Town02,Town03,Town04,Town05,Town10HD,Town12,Town13}
export ROUTE_SEED=${ROUTE_SEED:-42}
export ROUTE_ROOT=${ROUTE_ROOT:-"$GARAGE_ROOT/data"}
export ROUTE_SUBSET_XML=${ROUTE_SUBSET_XML:-"$WORK_ROOT/routes/pdm_lite_tesla_8town_3h_routes.xml"}
export ROUTE_SUBSET_META=${ROUTE_SUBSET_META:-"$WORK_ROOT/routes/pdm_lite_tesla_8town_3h_routes.json"}
export TOTAL_ROUTES=${TOTAL_ROUTES:-}
export REFRESH_ROUTE_SUBSET=${REFRESH_ROUTE_SUBSET:-0}
export TIMEOUT=${TIMEOUT:-600}
export RESUME=${RESUME:-1}
export REPETITION=${REPETITION:-0}
export TOWN=${TOWN:-pdm_lite_8town_tesla}
export DEBUG_CHALLENGE=${DEBUG_CHALLENGE:-0}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export SAVE_PATH="$OUTPUT_ROOT/data"
export CHECKPOINT_ENDPOINT=${CHECKPOINT_ENDPOINT:-"$OUTPUT_ROOT/results/pdm_lite_tesla_8town_3h_result.json"}
export DEBUG_CHECKPOINT=${DEBUG_CHECKPOINT:-"$OUTPUT_ROOT/results/pdm_lite_tesla_8town_3h_live.txt"}

mkdir -p "$OUTPUT_ROOT/data" "$OUTPUT_ROOT/results" "$WORK_ROOT/routes"

if [[ -z "$TOTAL_ROUTES" ]]; then
  TOTAL_ROUTES=$(python3 - <<PY
import math
print(int(math.ceil(float("$TARGET_HOURS") * 3600.0 / float("$AVG_ROUTE_SEC"))))
PY
)
fi

if [[ "$REFRESH_ROUTE_SUBSET" == "1" || ! -s "$ROUTE_SUBSET_XML" ]]; then
  python3 "$ADAPTER_ROOT/scripts/build_pdm_lite_route_subset.py" \
    --route-root "$ROUTE_ROOT" \
    --output "$ROUTE_SUBSET_XML" \
    --meta-output "$ROUTE_SUBSET_META" \
    --towns-csv "$TOWNS_CSV" \
    --target-hours "$TARGET_HOURS" \
    --avg-route-sec "$AVG_ROUTE_SEC" \
    --total-routes "$TOTAL_ROUTES" \
    --seed "$ROUTE_SEED"
fi

cat > "$OUTPUT_ROOT/collection_plan.json" <<EOF
{
  "dataset": "pdm_lite_tesla_8town_3h",
  "collection_style": "carla_garage_pdm_lite_route_expert",
  "ego_vehicle_model": "$EGO_VEHICLE_MODEL",
  "target_hours": $TARGET_HOURS,
  "avg_route_sec_assumption": $AVG_ROUTE_SEC,
  "total_routes": $TOTAL_ROUTES,
  "towns": "$TOWNS_CSV",
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
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$GARAGE_ROOT/leaderboard_autopilot:$GARAGE_ROOT/scenario_runner_autopilot:$GARAGE_ROOT/team_code:${PYTHONPATH:-}"
export REPETITIONS=1
export TEAM_AGENT="$GARAGE_ROOT/team_code/data_agent.py"
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES="$ROUTE_SUBSET_XML"
export TEAM_CONFIG="$ROUTE_SUBSET_XML"
export DATAGEN=1

echo "=== PDM-Lite Tesla route expert collection"
echo "GARAGE_ROOT=$GARAGE_ROOT"
echo "CARLA_ROOT=$CARLA_ROOT"
echo "EGO_VEHICLE_MODEL=$EGO_VEHICLE_MODEL"
echo "ROUTES=$ROUTES"
echo "TOTAL_ROUTES=$TOTAL_ROUTES"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "CHECKPOINT_ENDPOINT=$CHECKPOINT_ENDPOINT"

"$PY" leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py \
  --host "$HOST" \
  --port "$PORT" \
  --traffic-manager-port "$TM_PORT" \
  --traffic-manager-seed "$TM_SEED" \
  --routes "$ROUTE_SUBSET_XML" \
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
echo "=== saved_measurement_frames=$frames target_for_3h_at_4hz=43200"
