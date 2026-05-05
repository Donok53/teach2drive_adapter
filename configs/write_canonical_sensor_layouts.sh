#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-"$HOME/code/teach2drive_bootstrap/data/carla/town10_3cam_640x360"}

"$PY" -m teach2drive_adapter.write_sensor_layouts \
  --input-root "$DATA_ROOT" \
  --output-name sensor_layout.json
