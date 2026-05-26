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
if [[ ! -f "$LOAD_FILE" ]]; then
  echo "Missing pretrained/load checkpoint: $LOAD_FILE" >&2
  exit 1
fi

echo "=== full fine-tune TransFuser++"
echo "garage_root=$GARAGE_ROOT"
echo "train_root=$TRAIN_ROOT"
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
      --root_dir "$TRAIN_ROOT" \
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
