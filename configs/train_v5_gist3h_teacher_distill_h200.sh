#!/usr/bin/env bash
# v5: reprojection-teacher OUTPUT (trajectory) distillation on the 3h gist dataset.
# Reuses the previously-successful cached_visual_adapter recipe, but the teacher
# cache is built from GEOMETRICALLY REPROJECTED x=-1.5 views (NOT paired dual-rig
# data). Student visual encoder sees the real shifted cameras; targets are blended
# 0.75 toward the reprojected-view TF++ predictions.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=${ROOT:-/data/dataset/byeongjae}
WORK=${WORK:-$ROOT/runs/v5_teacher_distill}
export PY=${PY:-python}

STUDENT_CACHE=${STUDENT_CACHE:-$WORK/cache/student_prior.npz}
TEACHER_CACHE=${TEACHER_CACHE:-$WORK/cache/teacher_prior.npz}
INDEX=${INDEX:-$ROOT/datasets/t2d_gist3h_index.npz}
EPISODE_ROOT=${EPISODE_ROOT:-$ROOT/datasets/t2d_pdm_lite_front_triplet_shifted_3h}
GARAGE_ROOT=${GARAGE_ROOT:-/data/users/byeongjae/code/carla_garage}
OUT=${OUT:-$WORK/train}
mkdir -p "$OUT"

# --- recipe (matches train_tfpp_shifted_adapter_teacher_distill_v3) ---
EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-1.25e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
VAL_RATIO=${VAL_RATIO:-0.15}
SEED=${SEED:-41}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-15}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.0}
HIDDEN_DIM=${HIDDEN_DIM:-512}
LAYOUT_HIDDEN_DIM=${LAYOUT_HIDDEN_DIM:-128}
VISUAL_DIM=${VISUAL_DIM:-256}
VISUAL_TOKEN_DIM=${VISUAL_TOKEN_DIM:-192}
VISUAL_LAYERS=${VISUAL_LAYERS:-2}
VISUAL_HEADS=${VISUAL_HEADS:-4}
CAMERA_DROPOUT_PROB=${CAMERA_DROPOUT_PROB:-0.0}
FRONT_CAMERA_DROPOUT_PROB=${FRONT_CAMERA_DROPOUT_PROB:-0.0}
MOVING_SPEED_THRESHOLD=${MOVING_SPEED_THRESHOLD:-1.0}
MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-2.0}
STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-0.50}
TEACHER_TARGET_BLEND=${TEACHER_TARGET_BLEND:-0.75}
XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-1.0}
YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.05}
SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.45}
SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.05}
SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.02}
SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.08}
SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-1.0}
PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.01}
STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.005}
STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.01}
STOP_POSITIVE_LOSS_SCALE=${STOP_POSITIVE_LOSS_SCALE:-0.25}
STOP_NEGATIVE_LOSS_SCALE=${STOP_NEGATIVE_LOSS_SCALE:-2.0}
STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-4}
TEACHER_TRAJ_BLEND=${TEACHER_TRAJ_BLEND:-$TEACHER_TARGET_BLEND}
TEACHER_SPEED_TARGET_BLEND=${TEACHER_SPEED_TARGET_BLEND:-$TEACHER_TARGET_BLEND}
TEACHER_STOP_TARGET_BLEND=${TEACHER_STOP_TARGET_BLEND:-$TEACHER_TARGET_BLEND}

echo "=== v5 reprojection-teacher distillation ==="
echo "student=$STUDENT_CACHE"
echo "teacher=$TEACHER_CACHE"
echo "out=$OUT"

PYTHONUNBUFFERED=1 "$PY" -m teach2drive_adapter.train_transfuserpp_cached_visual_adapter \
  --cache "$STUDENT_CACHE" \
  --teacher-cache "$TEACHER_CACHE" \
  --index "$INDEX" \
  --episode-root-override "$EPISODE_ROOT" \
  --out-dir "$OUT" \
  --cameras left,front,right \
  --use-raw-layout \
  --image-size 320 180 \
  --lidar-size 128 \
  --epochs "$EPOCHS" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --hidden-dim "$HIDDEN_DIM" \
  --layout-hidden-dim "$LAYOUT_HIDDEN_DIM" \
  --visual-dim "$VISUAL_DIM" \
  --visual-token-dim "$VISUAL_TOKEN_DIM" \
  --visual-layers "$VISUAL_LAYERS" \
  --visual-heads "$VISUAL_HEADS" \
  --camera-dropout-prob "$CAMERA_DROPOUT_PROB" \
  --front-camera-dropout-prob "$FRONT_CAMERA_DROPOUT_PROB" \
  --moving-speed-threshold "$MOVING_SPEED_THRESHOLD" \
  --moving-sample-weight "$MOVING_SAMPLE_WEIGHT" \
  --stopped-sample-weight "$STOPPED_SAMPLE_WEIGHT" \
  --teacher-target-blend "$TEACHER_TARGET_BLEND" \
  --xy-loss-weight "$XY_LOSS_WEIGHT" \
  --yaw-loss-weight "$YAW_LOSS_WEIGHT" \
  --speed-loss-weight "$SPEED_LOSS_WEIGHT" \
  --speed-teacher-blend "$SPEED_TEACHER_BLEND" \
  --speed-distill-loss-weight "$SPEED_DISTILL_LOSS_WEIGHT" \
  --speed-floor-loss-weight "$SPEED_FLOOR_LOSS_WEIGHT" \
  --speed-floor-mps "$SPEED_FLOOR_MPS" \
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
  --stop-loss-weight "$STOP_LOSS_WEIGHT" \
  --stop-state-loss-weight "$STOP_STATE_LOSS_WEIGHT" \
  --stop-positive-loss-scale "$STOP_POSITIVE_LOSS_SCALE" \
  --stop-negative-loss-scale "$STOP_NEGATIVE_LOSS_SCALE" \
  --stop-loss-after-epoch "$STOP_LOSS_AFTER_EPOCH" \
  --teacher-traj-blend "$TEACHER_TRAJ_BLEND" \
  --teacher-speed-target-blend "$TEACHER_SPEED_TARGET_BLEND" \
  --teacher-stop-target-blend "$TEACHER_STOP_TARGET_BLEND" \
  --data-parallel \
  2>&1 | tee "$OUT/train.log"
