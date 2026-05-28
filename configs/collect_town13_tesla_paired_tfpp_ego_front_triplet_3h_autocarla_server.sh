#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# One-command wrapper for Town13-only domain shift collection.
# This delegates CARLA startup/reuse to the shared autocarla wrapper, then
# runs collect_town13_tesla_paired_tfpp_ego_front_triplet_3h.sh.

export COLLECT_CONFIG=${COLLECT_CONFIG:-configs/collect_town13_tesla_paired_tfpp_ego_front_triplet_3h.sh}
export CARLA_LOG=${CARLA_LOG:-"$HOME/teach2drive/logs/carla_town13_tesla_collect.log"}

exec bash configs/collect_vehicle_b_town13_paired_tfpp_ego_front_triplet_3h_autocarla_server.sh
