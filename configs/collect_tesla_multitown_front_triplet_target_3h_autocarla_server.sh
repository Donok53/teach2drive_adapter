#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Collect a total 3h small expert dataset across benchmark-aligned towns.
# Default split: Longest6 towns, 36 episodes x 300s = 3 simulated hours.
# Target-only: front_triplet_shifted is the only sensor profile collected.

export CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
export TOWN_PRESET=${TOWN_PRESET:-longest6}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_${TOWN_PRESET}_front_triplet_target_3h"}
export COLLECT_CONFIG=${COLLECT_CONFIG:-configs/collect_tesla_front_triplet_target_3h.sh}
export CARLA_LOG=${CARLA_LOG:-"$HOME/teach2drive/logs/carla_multitown_tesla_front_triplet_collect.log"}

export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-2000}
export VEHICLE_FILTER=${VEHICLE_FILTER:-vehicle.tesla.model3}
export PROFILES=front_triplet_shifted
export PRIMARY_PROFILE=front_triplet_shifted
export COLLECTION_MODE=paired
export EPISODE_SEC=${EPISODE_SEC:-300}
export TOTAL_EPISODES=${TOTAL_EPISODES:-36}
export OVERWRITE=${OVERWRITE:-0}
export TRAFFIC_SCHEDULE=${TRAFFIC_SCHEDULE:-vehicle_b_mixed}
export TRAFFIC_VEHICLES=${TRAFFIC_VEHICLES:-60}
export FAIL_ON_INVALID_MOTION=${FAIL_ON_INVALID_MOTION:-1}
export CONTINUE_ON_TOWN_FAILURE=${CONTINUE_ON_TOWN_FAILURE:-1}

if [[ -z "${TOWNS_CSV:-}" ]]; then
  case "$TOWN_PRESET" in
    longest6)
      TOWNS_CSV="Town01,Town02,Town03,Town04,Town05,Town06"
      ;;
    bench2drive)
      TOWNS_CSV="Town01,Town02,Town03,Town04,Town05,Town06,Town07,Town10HD,Town11,Town12,Town13,Town15"
      ;;
    *)
      echo "Unknown TOWN_PRESET=$TOWN_PRESET. Set TOWNS_CSV explicitly or use longest6/bench2drive." >&2
      exit 2
      ;;
  esac
fi
IFS=',' read -r -a TOWNS <<< "$TOWNS_CSV"

mkdir -p "$(dirname "$CARLA_LOG")" "$OUTPUT_ROOT"
FAILED_TOWNS_LOG="$OUTPUT_ROOT/failed_towns.jsonl"
: > "$FAILED_TOWNS_LOG"

stop_carla() {
  if pgrep -f CarlaUE4 >/dev/null 2>&1; then
    pkill -TERM -f CarlaUE4 2>/dev/null || true
    sleep 3
    pkill -KILL -f CarlaUE4 2>/dev/null || true
  fi
}

TOWN_COUNT=0
for town in "${TOWNS[@]}"; do
  town="${town//[[:space:]]/}"
  [[ -z "$town" ]] && continue
  TOWN_COUNT=$((TOWN_COUNT + 1))
done
if [[ "$TOWN_COUNT" -le 0 ]]; then
  echo "No towns selected. TOWNS_CSV=$TOWNS_CSV" >&2
  exit 2
fi
if [[ "$TOTAL_EPISODES" -lt "$TOWN_COUNT" ]]; then
  echo "TOTAL_EPISODES=$TOTAL_EPISODES is smaller than selected towns=$TOWN_COUNT" >&2
  exit 2
fi

episode_start=${START_EPISODE_INDEX:-0}
episode_cursor=$episode_start
town_offset=0
completed_episodes=0
failed_towns=""
base_episodes=$((TOTAL_EPISODES / TOWN_COUNT))
extra_episodes=$((TOTAL_EPISODES % TOWN_COUNT))
for town in "${TOWNS[@]}"; do
  town="${town//[[:space:]]/}"
  [[ -z "$town" ]] && continue

  episodes_for_town=$base_episodes
  if [[ "$town_offset" -lt "$extra_episodes" ]]; then
    episodes_for_town=$((episodes_for_town + 1))
  fi
  export MAP="$town"
  export START_EPISODE_INDEX="$episode_cursor"
  export EPISODES="$episodes_for_town"
  export TRAFFIC_SCHEDULE_SEED=$((31 + town_offset))

  echo "=== collect target-only expert data: map=$MAP episodes=$EPISODES start=$START_EPISODE_INDEX output=$OUTPUT_ROOT"
  if bash configs/collect_autocarla_server.sh; then
    completed_episodes=$((completed_episodes + episodes_for_town))
  else
    status=$?
    echo "{\"town\":\"$MAP\",\"episodes\":$EPISODES,\"start_episode_index\":$START_EPISODE_INDEX,\"exit_status\":$status}" >> "$FAILED_TOWNS_LOG"
    failed_towns="${failed_towns}${failed_towns:+,}$MAP"
    echo "=== WARNING: collection failed for map=$MAP status=$status"
    stop_carla
    if [[ "$CONTINUE_ON_TOWN_FAILURE" != "1" ]]; then
      exit "$status"
    fi
  fi

  episode_cursor=$((episode_cursor + episodes_for_town))
  town_offset=$((town_offset + 1))
done

cat > "$OUTPUT_ROOT/multitown_plan.json" <<EOF
{
  "dataset": "t2d_tesla_${TOWN_PRESET}_front_triplet_target_3h",
  "town_preset": "$TOWN_PRESET",
  "profiles": ["front_triplet_shifted"],
  "vehicle_filter": "$VEHICLE_FILTER",
  "towns": "$TOWNS_CSV",
  "total_episodes": $TOTAL_EPISODES,
  "completed_episodes": $completed_episodes,
  "failed_towns": "$failed_towns",
  "continue_on_town_failure": $CONTINUE_ON_TOWN_FAILURE,
  "episode_sec": $EPISODE_SEC,
  "total_hours": $(python - <<PY
print(int("$TOTAL_EPISODES") * float("$EPISODE_SEC") / 3600.0)
PY
)
}
EOF
