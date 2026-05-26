#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Full TransFuser++ fine-tune baseline.
#
# This is the "does full-model supervised fine-tuning beat the adapter?"
# comparison run.  It exports a Teach2Drive sensor profile into the minimal
# CARLA Garage route-folder format, then calls carla_garage/team_code/train.py
# with all TransFuser++ parameters trainable.
#
# The exported dataset intentionally disables semantic/depth/box auxiliary
# losses because the Teach2Drive paired-profile logs do not contain the full
# CARLA Garage supervision stack.

PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}
WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_full_finetune_front_triplet_shifted_v1"}
GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}
CARLA_ROOT=${CARLA_ROOT:-"$HOME/dataset/byeongjae/carla-simulator"}

if [[ ! -d "$GARAGE_ROOT" && -d "$HOME/code/carla_garage" ]]; then
  GARAGE_ROOT="$HOME/code/carla_garage"
fi
if [[ ! -d "$TEAM_CONFIG" && -d "$HOME/code/checkpoints/transfuserpp/pretrained_models/all_towns" ]]; then
  TEAM_CONFIG="$HOME/code/checkpoints/transfuserpp/pretrained_models/all_towns"
fi

PROFILE=${PROFILE:-front_triplet_shifted}
TFPP_CAMERA=${TFPP_CAMERA:-front}
TOWN=${TOWN:-Town13}
SCENARIO_NAME=${SCENARIO_NAME:-"${PROFILE}_minimal_tfpp_train"}
GARAGE_DATA_ROOT=${GARAGE_DATA_ROOT:-"$WORK_ROOT/carla_garage_dataset"}
TRAIN_ROOT=${TRAIN_ROOT:-"$GARAGE_DATA_ROOT/$SCENARIO_NAME"}
FILTERED_DATA_ROOT=${FILTERED_DATA_ROOT:-"$WORK_ROOT/carla_garage_dataset_filtered"}
FILTERED_TRAIN_ROOT=${FILTERED_TRAIN_ROOT:-"$FILTERED_DATA_ROOT/$SCENARIO_NAME"}
RUN_ID=${RUN_ID:-"tfpp_full_finetune_${PROFILE}_v1"}
LOGDIR=${LOGDIR:-"$WORK_ROOT/models"}

LOAD_FILE=${LOAD_FILE:-"$TEAM_CONFIG/model_0030_0.pth"}
EPOCHS=${EPOCHS:-15}
LR=${LR:-3e-5}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-8}
CPU_CORES=${CPU_CORES:-16}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
NUM_REPETITIONS=${NUM_REPETITIONS:-1}
SEED=${SEED:-41}
OVERWRITE_EXPORT=${OVERWRITE_EXPORT:-0}
SKIP_INVALID_MOTION=${SKIP_INVALID_MOTION:-1}
CONVERT_NPZ_LIDAR=${CONVERT_NPZ_LIDAR:-1}
ZERO_REDUNDANCY_OPTIMIZER=${ZERO_REDUNDANCY_OPTIMIZER:-1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
SETTING=${SETTING:-all}

mkdir -p "$WORK_ROOT" "$GARAGE_DATA_ROOT" "$LOGDIR"

PYTHON_SHIM_DIR=${PYTHON_SHIM_DIR:-"$WORK_ROOT/python_shims"}
mkdir -p "$PYTHON_SHIM_DIR"

if [[ -d "$CARLA_ROOT/PythonAPI/carla" ]]; then
  export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-*.egg:${PYTHONPATH:-}"
fi

if ! "$PY" -c "import carla" >/dev/null 2>&1; then
  cat > "$PYTHON_SHIM_DIR/carla.py" <<'PY'
"""Small offline CARLA shim for CARLA Garage training imports.

CARLA Garage's train.py imports carla for configuration constants and helper
classes even when no simulator is used.  Some server environments only ship a
Python 3.7 CARLA egg, while training runs in Python 3.10.  This shim is enough
for offline supervised training and intentionally does not implement simulator
client behavior.
"""

import math


class _EnumValue:
    def __init__(self, name, value=0):
        self.name = name
        self.value = int(value)

    def __int__(self):
        return self.value

    def __or__(self, other):
        return _EnumValue(f"{self.name}|{getattr(other, 'name', other)}", self.value | int(other))

    def __repr__(self):
        return f"carla.{self.name}"


class _EnumNamespace:
    def __init__(self, **values):
        self._values = values

    def __getattr__(self, name):
        value = self._values.setdefault(name, _EnumValue(name, len(self._values) + 1))
        return value


LaneType = _EnumNamespace(Driving=_EnumValue("Driving", 1), Shoulder=_EnumValue("Shoulder", 2), Parking=_EnumValue("Parking", 4), Sidewalk=_EnumValue("Sidewalk", 8), Biking=_EnumValue("Biking", 16))
LaneChange = _EnumNamespace(None_=_EnumValue("None", 0), Right=_EnumValue("Right", 1), Left=_EnumValue("Left", 2), Both=_EnumValue("Both", 3))
LaneMarkingType = _EnumNamespace(NONE=_EnumValue("NONE", 0), Broken=_EnumValue("Broken", 1), Solid=_EnumValue("Solid", 2), SolidBroken=_EnumValue("SolidBroken", 3), BrokenSolid=_EnumValue("BrokenSolid", 4), BrokenBroken=_EnumValue("BrokenBroken", 5), SolidSolid=_EnumValue("SolidSolid", 6))
LaneMarkingColor = _EnumNamespace(Other=_EnumValue("Other", 0), White=_EnumValue("White", 1), Yellow=_EnumValue("Yellow", 2))
TrafficLightState = _EnumNamespace(Red=_EnumValue("Red", 0), Yellow=_EnumValue("Yellow", 1), Green=_EnumValue("Green", 2), Off=_EnumValue("Off", 3), Unknown=_EnumValue("Unknown", 4))
LandmarkType = _EnumNamespace(MaximumSpeed=_EnumValue("MaximumSpeed", 0))


class libcarla:
    TrafficLightState = TrafficLightState


class Color:
    def __init__(self, r=0, g=0, b=0, a=255):
        self.r = r
        self.g = g
        self.b = b
        self.a = a


class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        if hasattr(x, "x") and y == 0.0 and z == 0.0:
            self.x = float(x.x)
            self.y = float(x.y)
            self.z = float(getattr(x, "z", 0.0))
        else:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    def __add__(self, other):
        return self.__class__(self.x + other.x, self.y + other.y, self.z + getattr(other, "z", 0.0))

    def __sub__(self, other):
        return self.__class__(self.x - other.x, self.y - other.y, self.z - getattr(other, "z", 0.0))

    def __mul__(self, value):
        return self.__class__(self.x * value, self.y * value, self.z * value)

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


class Vector2D:
    def __init__(self, x=0.0, y=0.0):
        if hasattr(x, "x") and y == 0.0:
            self.x = float(x.x)
            self.y = float(x.y)
        else:
            self.x = float(x)
            self.y = float(y)


class Location(Vector3D):
    def distance(self, other):
        return (self - other).length()


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location if location is not None else Location()
        self.rotation = rotation if rotation is not None else Rotation()

    def transform(self, point):
        return self.location + point


class BoundingBox:
    def __init__(self, location=None, extent=None):
        self.location = location if location is not None else Location()
        self.extent = extent if extent is not None else Vector3D()
        self.rotation = Rotation()

    def get_world_vertices(self, transform):
        return []


class VehicleControl:
    def __init__(self):
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0


class VehicleLightState:
    Position = _EnumValue("Position", 1)
    LowBeam = _EnumValue("LowBeam", 2)


class WorldSettings:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Client:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("The offline CARLA shim cannot create simulator clients.")
PY
  export PYTHONPATH="$PYTHON_SHIM_DIR:${PYTHONPATH:-}"
  echo "=== using offline carla shim: $PYTHON_SHIM_DIR/carla.py"
fi

EXPORT_ARGS=()
if [[ "$OVERWRITE_EXPORT" == "1" || "$OVERWRITE_EXPORT" == "true" || "$OVERWRITE_EXPORT" == "TRUE" ]]; then
  EXPORT_ARGS+=(--overwrite)
fi
if [[ "$SKIP_INVALID_MOTION" == "1" || "$SKIP_INVALID_MOTION" == "true" || "$SKIP_INVALID_MOTION" == "TRUE" ]]; then
  EXPORT_ARGS+=(--skip-invalid-motion)
fi
if [[ "$CONVERT_NPZ_LIDAR" == "0" || "$CONVERT_NPZ_LIDAR" == "false" || "$CONVERT_NPZ_LIDAR" == "FALSE" ]]; then
  EXPORT_ARGS+=(--no-convert-npz-lidar)
fi

echo "=== export Teach2Drive profile to CARLA Garage format"
"$PY" -m teach2drive_adapter.export_carla_garage_profile_dataset \
  --input-root "$DATA_ROOT" \
  --output-root "$GARAGE_DATA_ROOT" \
  --scenario-name "$SCENARIO_NAME" \
  --profile "$PROFILE" \
  --camera "$TFPP_CAMERA" \
  --town "$TOWN" \
  --prefix-index \
  "${EXPORT_ARGS[@]}"

if [[ ! -d "$TRAIN_ROOT" ]]; then
  echo "Missing exported train root: $TRAIN_ROOT" >&2
  exit 1
fi

echo "=== build filtered CARLA Garage train root"
FILTERED_SOURCE_ROOT="$TRAIN_ROOT" FILTERED_TARGET_ROOT="$FILTERED_TRAIN_ROOT" "$PY" - <<'PY'
import os
import re
import shutil
from pathlib import Path

source = Path(os.environ["FILTERED_SOURCE_ROOT"]).expanduser()
target = Path(os.environ["FILTERED_TARGET_ROOT"]).expanduser()
if target.exists() or target.is_symlink():
    shutil.rmtree(target)
target.mkdir(parents=True, exist_ok=True)

route_names = []
skipped = []
for child in sorted(source.iterdir()):
    if not child.is_dir():
        continue
    if re.search(r"_Rep\d+$", child.name) is None:
        skipped.append(child.name)
        continue
    if not (child / "results.json.gz").is_file():
        skipped.append(child.name)
        continue
    os.symlink(child, target / child.name, target_is_directory=True)
    route_names.append(child.name)

if not route_names:
    raise RuntimeError(f"No CARLA Garage route directories found in {source}")
print(
    {
        "source": str(source),
        "target": str(target),
        "routes": len(route_names),
        "skipped_dirs": skipped,
    },
    flush=True,
)
PY

if [[ ! -f "$LOAD_FILE" ]]; then
  echo "Missing pretrained/load checkpoint: $LOAD_FILE" >&2
  exit 1
fi

echo "=== full fine-tune TransFuser++"
echo "garage_root=$GARAGE_ROOT"
echo "train_root=$FILTERED_TRAIN_ROOT"
echo "root_dir=$FILTERED_DATA_ROOT"
echo "logdir=$LOGDIR/$RUN_ID"
echo "load_file=$LOAD_FILE"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

(
  cd "$GARAGE_ROOT/team_code"
  PYTHONPATH="$GARAGE_ROOT/team_code:${PYTHONPATH:-}" \
  torchrun \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    --max_restarts=0 \
    --rdzv_id="${RDZV_ID:-$RANDOM$RANDOM}" \
    --rdzv_backend=c10d \
    train.py \
      --logdir "$LOGDIR" \
      --root_dir "$FILTERED_DATA_ROOT" \
      --id "$RUN_ID" \
      --load_file "$LOAD_FILE" \
      --continue_epoch 0 \
      --setting "$SETTING" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      --batch_size "$BATCH_SIZE_PER_GPU" \
      --cpu_cores "$CPU_CORES" \
      --num_repetitions "$NUM_REPETITIONS" \
      --zero_redundancy_optimizer "$ZERO_REDUNDANCY_OPTIMIZER" \
      --weight_decay "$WEIGHT_DECAY" \
      --use_semantic 0 \
      --use_bev_semantic 0 \
      --use_depth 0 \
      --detect_boxes 0 \
      --augment 0 \
      --use_color_aug 0 \
      --use_controller_input_prediction 1 \
      --use_wp_gru 0 \
      --lidar_seq_len 1 \
      --sync_batch_norm 0 \
      --compile 0 \
      --seed "$SEED"
)
