#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Collect a total 3h small expert dataset across multiple towns.
# Default split: 4 towns x 9 episodes x 300s = 3 simulated hours.
# Target-only: front_triplet_shifted is the only sensor profile collected.

export CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}
export OUTPUT_ROOT=${OUTPUT_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_multitown_front_triplet_target_3h"}
export COLLECT_CONFIG=${COLLECT_CONFIG:-configs/collect_tesla_front_triplet_target_3h.sh}
export CARLA_LOG=${CARLA_LOG:-"$HOME/teach2drive/logs/carla_multitown_tesla_front_triplet_collect.log"}

export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-2000}
export VEHICLE_FILTER=${VEHICLE_FILTER:-vehicle.tesla.model3}
export PROFILES=front_triplet_shifted
export PRIMARY_PROFILE=front_triplet_shifted
export COLLECTION_MODE=paired
export EPISODE_SEC=${EPISODE_SEC:-300}
export EPISODES_PER_TOWN=${EPISODES_PER_TOWN:-9}
export OVERWRITE=${OVERWRITE:-0}
export TRAFFIC_SCHEDULE=${TRAFFIC_SCHEDULE:-vehicle_b_mixed}
export TRAFFIC_VEHICLES=${TRAFFIC_VEHICLES:-60}
export FAIL_ON_INVALID_MOTION=${FAIL_ON_INVALID_MOTION:-1}

TOWNS_CSV=${TOWNS_CSV:-Town03,Town05,Town06,Town10HD_Opt}
IFS=',' read -r -a TOWNS <<< "$TOWNS_CSV"

mkdir -p "$(dirname "$CARLA_LOG")" "$OUTPUT_ROOT"

episode_start=${START_EPISODE_INDEX:-0}
town_offset=0
for town in "${TOWNS[@]}"; do
  town="${town//[[:space:]]/}"
  [[ -z "$town" ]] && continue

  start_index=$((episode_start + town_offset * EPISODES_PER_TOWN))
  export MAP="$town"
  export START_EPISODE_INDEX="$start_index"
  export EPISODES="$EPISODES_PER_TOWN"
  export TRAFFIC_SCHEDULE_SEED=$((31 + town_offset))

  echo "=== collect target-only expert data: map=$MAP episodes=$EPISODES start=$START_EPISODE_INDEX output=$OUTPUT_ROOT"
  bash configs/collect_autocarla_server.sh

  town_offset=$((town_offset + 1))
done

cat > "$OUTPUT_ROOT/multitown_plan.json" <<EOF
{
  "dataset": "t2d_tesla_multitown_front_triplet_target_3h",
  "profiles": ["front_triplet_shifted"],
  "vehicle_filter": "$VEHICLE_FILTER",
  "towns": "$TOWNS_CSV",
  "episodes_per_town": $EPISODES_PER_TOWN,
  "episode_sec": $EPISODE_SEC,
  "total_hours": $(python - <<PY
episodes = int("$EPISODES_PER_TOWN") * len([t for t in "$TOWNS_CSV".split(",") if t.strip()])
print(episodes * float("$EPISODE_SEC") / 3600.0)
PY
)
}
EOF
