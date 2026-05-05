#!/usr/bin/env bash
set -euo pipefail

# Install the CARLA Garage TransFuser++ code and official pretrained weights.
# This script intentionally does not install the conda environment by default,
# because local and remote GPU nodes usually need different CUDA images.

ROOT=${ROOT:-"$HOME/code"}
GARAGE_DIR=${GARAGE_DIR:-"$ROOT/carla_garage"}
WEIGHT_DIR=${WEIGHT_DIR:-"$ROOT/checkpoints/transfuserpp"}
CARLA_ROOT=${CARLA_ROOT:-"$HOME/carla-simulator"}

MODEL_ZIP_URL=${MODEL_ZIP_URL:-"https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip"}
MODEL_ZIP=${MODEL_ZIP:-"$WEIGHT_DIR/pretrained_models.zip"}

mkdir -p "$ROOT" "$WEIGHT_DIR"

if [[ ! -d "$GARAGE_DIR/.git" ]]; then
  git clone https://github.com/autonomousvision/carla_garage.git "$GARAGE_DIR"
fi

git -C "$GARAGE_DIR" fetch origin leaderboard_2
git -C "$GARAGE_DIR" checkout leaderboard_2

echo "[teach2drive] CARLA Garage path: $GARAGE_DIR"
echo "[teach2drive] Weight path:       $WEIGHT_DIR"
echo "[teach2drive] CARLA root:        $CARLA_ROOT"

if [[ ! -f "$MODEL_ZIP" ]]; then
  echo "[teach2drive] Downloading TransFuser++ pretrained models."
  echo "[teach2drive] File is about 2.6 GB; resume is enabled when possible."
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 8 -s 8 -d "$WEIGHT_DIR" -o "$(basename "$MODEL_ZIP")" "$MODEL_ZIP_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$MODEL_ZIP" "$MODEL_ZIP_URL"
  else
    curl -L -C - -o "$MODEL_ZIP" "$MODEL_ZIP_URL"
  fi
fi

if [[ ! -d "$WEIGHT_DIR/pretrained_models" ]]; then
  echo "[teach2drive] Unzipping model archive."
  unzip -q "$MODEL_ZIP" -d "$WEIGHT_DIR"
fi

cat <<EOF

[teach2drive] Done.

Add this before running CARLA Garage code:

export CARLA_ROOT="$CARLA_ROOT"
export WORK_DIR="$GARAGE_DIR"
export SCENARIO_RUNNER_ROOT="\$WORK_DIR/scenario_runner"
export LEADERBOARD_ROOT="\$WORK_DIR/leaderboard"
export PYTHONPATH="\$CARLA_ROOT/PythonAPI/carla/:\$SCENARIO_RUNNER_ROOT:\$LEADERBOARD_ROOT:\$PYTHONPATH"

Recommended next checks:

find "$WEIGHT_DIR" -maxdepth 3 -type f \( -name 'config.json' -o -name 'args.txt' -o -name '*.pth' \) | sort

EOF
