#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export ROOT=${ROOT:-"$HOME/code"}
export GARAGE_DIR=${GARAGE_DIR:-"$ROOT/carla_garage"}
export WEIGHT_DIR=${WEIGHT_DIR:-"$ROOT/checkpoints/transfuserpp"}
export CARLA_ROOT=${CARLA_ROOT:-"$HOME/carla-simulator"}

bash scripts/install_transfuserpp.sh
