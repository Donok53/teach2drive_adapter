#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# B-vehicle expert-only preset.
# This is intentionally not a canonical TransFuser++ teacher-distillation run:
# the pretrained model supplies an input prior/cache, while the supervision
# target is the expert trajectory/speed/stop labels collected on vehicle B.

PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_vehicle_b_front_triplet_mixed_3h"}
WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_vehicle_b_front_triplet_expert_v1"}
BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-"$HOME/teach2drive/workspace/teach2drive_bootstrap"}
GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}
TFPP_CHECKPOINT=${TFPP_CHECKPOINT:-""}

PROFILE=${PROFILE:-front_triplet_shifted}
CAMERAS=${CAMERAS:-left,front,right}
VIEW_ROOT=${VIEW_ROOT:-"$WORK_ROOT/profile_views"}
PROFILE_VIEW=${PROFILE_VIEW:-"$VIEW_ROOT/$PROFILE"}
INDEX_DIR=${INDEX_DIR:-"$WORK_ROOT/indexes"}
INDEX=${INDEX:-"$INDEX_DIR/${PROFILE}_index.npz"}
CACHE=${CACHE:-"$WORK_ROOT/cache/${PROFILE}_prior_cache.npz"}
OUT=${OUT:-"$WORK_ROOT/train_${PROFILE}_expert_v1"}

AUGMENTATIONS=${AUGMENTATIONS:-0}
EPOCHS=${EPOCHS:-80}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-15}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.0}
BATCH_SIZE=${BATCH_SIZE:-128}
CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
CACHE_WORKERS=${CACHE_WORKERS:-16}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
VAL_RATIO=${VAL_RATIO:-0.15}
SEED=${SEED:-41}
OVERWRITE=${OVERWRITE:-0}
EXPORT_OVERWRITE=${EXPORT_OVERWRITE:-$OVERWRITE}
INDEX_OVERWRITE=${INDEX_OVERWRITE:-$OVERWRITE}
DATA_PARALLEL=${DATA_PARALLEL:-1}
SKIP_INVALID_MOTION=${SKIP_INVALID_MOTION:-1}

# Capacity: larger than v3, but still small enough for the 3-hour dataset.
HIDDEN_DIM=${HIDDEN_DIM:-768}
LAYOUT_HIDDEN_DIM=${LAYOUT_HIDDEN_DIM:-192}
VISUAL_DIM=${VISUAL_DIM:-384}
VISUAL_TOKEN_DIM=${VISUAL_TOKEN_DIM:-256}
VISUAL_LAYERS=${VISUAL_LAYERS:-4}
VISUAL_HEADS=${VISUAL_HEADS:-8}

# Force the adapter to use side views without making training too noisy.
CAMERA_DROPOUT_PROB=${CAMERA_DROPOUT_PROB:-0.02}
FRONT_CAMERA_DROPOUT_PROB=${FRONT_CAMERA_DROPOUT_PROB:-0.10}

# Expert imitation, with more emphasis on moving/hazard frames.
MOVING_SPEED_THRESHOLD=${MOVING_SPEED_THRESHOLD:-1.0}
MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.35}
STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-0.85}
HAZARD_STOP_REASONS=${HAZARD_STOP_REASONS:-"traffic_light,stop_sign,front_vehicle,junction_yield"}
HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-3.0}

# No teacher-follow target blending. A small prior loss keeps the residual
# adapter from drifting wildly away from pretrained TransFuser++ behavior.
TEACHER_TARGET_BLEND=${TEACHER_TARGET_BLEND:-0.0}
TEACHER_TRAJ_BLEND=${TEACHER_TRAJ_BLEND:-0.0}
TEACHER_SPEED_TARGET_BLEND=${TEACHER_SPEED_TARGET_BLEND:-0.0}
TEACHER_STOP_TARGET_BLEND=${TEACHER_STOP_TARGET_BLEND:-0.0}
HAZARD_SPEED_TARGET_BLEND=${HAZARD_SPEED_TARGET_BLEND:--1.0}
SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.0}
SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.0}
PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.03}

XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-1.0}
YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.05}
SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-1.0}
SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.01}
SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-0.3}
SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.08}
SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.03}
TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.05}
TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.02}

STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.08}
STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.15}
STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.08}
STOP_POSITIVE_LOSS_SCALE=${STOP_POSITIVE_LOSS_SCALE:-1.3}
STOP_NEGATIVE_LOSS_SCALE=${STOP_NEGATIVE_LOSS_SCALE:-1.0}
STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-1}

mkdir -p "$WORK_ROOT" "$INDEX_DIR" "$(dirname "$CACHE")" "$OUT"

EXPORT_ARGS=()
if [[ "$EXPORT_OVERWRITE" == "1" ]]; then
  EXPORT_ARGS+=(--overwrite)
fi
if [[ "$SKIP_INVALID_MOTION" == "1" || "$SKIP_INVALID_MOTION" == "true" || "$SKIP_INVALID_MOTION" == "TRUE" ]]; then
  EXPORT_ARGS+=(--skip-invalid-motion)
fi

echo "=== export $PROFILE profile view"
"$PY" -m teach2drive_adapter.export_paired_profile_view \
  --input-root "$DATA_ROOT" \
  --output-root "$PROFILE_VIEW" \
  --profile "$PROFILE" \
  --require-cameras "$CAMERAS" \
  "${EXPORT_ARGS[@]}"

if [[ -f "$INDEX" && "$INDEX_OVERWRITE" != "1" ]]; then
  echo "=== reuse index $INDEX"
else
  echo "=== build index $INDEX cameras=$CAMERAS"
  (
    cd "$BOOTSTRAP_ROOT"
    PYTHONPATH="$BOOTSTRAP_ROOT:${PYTHONPATH:-}" "$PY" -m teach2drive.token_dataset \
      --input-root "$PROFILE_VIEW" \
      --output "$INDEX" \
      --cameras "$CAMERAS" \
      --augmentations "$AUGMENTATIONS" \
      --pseudo-label-name "__missing_pseudo_labels__.jsonl" \
      --seed "$SEED"
  )
fi

CACHE_ARGS=()
if [[ "$DATA_PARALLEL" == "1" ]]; then
  CACHE_ARGS+=(--data-parallel)
fi
if [[ -n "$TFPP_CHECKPOINT" ]]; then
  CACHE_ARGS+=(--checkpoint "$TFPP_CHECKPOINT")
fi

if [[ -f "$CACHE" && "$OVERWRITE" != "1" ]]; then
  echo "=== reuse B-rig TransFuser++ prior cache $CACHE"
else
  echo "=== cache B-rig TransFuser++ prior"
  "$PY" -m teach2drive_adapter.cache_transfuserpp_prior \
    --index "$INDEX" \
    --output "$CACHE" \
    --garage-root "$GARAGE_ROOT" \
    --team-config "$TEAM_CONFIG" \
    --episode-root-override "$PROFILE_VIEW" \
    --cameras "$CAMERAS" \
    --tfpp-camera front \
    --command-mode target_angle \
    --image-size 640 360 \
    --lidar-size 128 \
    --batch-size "$CACHE_BATCH_SIZE" \
    --num-workers "$CACHE_WORKERS" \
    "${CACHE_ARGS[@]}"
fi

TRAIN_ARGS=()
if [[ "$DATA_PARALLEL" == "1" ]]; then
  TRAIN_ARGS+=(--data-parallel)
fi

echo "=== train B-vehicle expert adapter"
PYTHONUNBUFFERED=1 "$PY" -m teach2drive_adapter.train_transfuserpp_cached_visual_adapter \
  --cache "$CACHE" \
  --index "$INDEX" \
  --episode-root-override "$PROFILE_VIEW" \
  --out-dir "$OUT" \
  --cameras "$CAMERAS" \
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
  --teacher-traj-blend "$TEACHER_TRAJ_BLEND" \
  --teacher-speed-target-blend "$TEACHER_SPEED_TARGET_BLEND" \
  --teacher-stop-target-blend "$TEACHER_STOP_TARGET_BLEND" \
  --hazard-stop-reasons "$HAZARD_STOP_REASONS" \
  --hazard-speed-target-blend "$HAZARD_SPEED_TARGET_BLEND" \
  --hazard-sample-weight "$HAZARD_SAMPLE_WEIGHT" \
  --xy-loss-weight "$XY_LOSS_WEIGHT" \
  --yaw-loss-weight "$YAW_LOSS_WEIGHT" \
  --speed-loss-weight "$SPEED_LOSS_WEIGHT" \
  --speed-teacher-blend "$SPEED_TEACHER_BLEND" \
  --speed-distill-loss-weight "$SPEED_DISTILL_LOSS_WEIGHT" \
  --speed-floor-loss-weight "$SPEED_FLOOR_LOSS_WEIGHT" \
  --speed-floor-mps "$SPEED_FLOOR_MPS" \
  --speed-delta-loss-weight "$SPEED_DELTA_LOSS_WEIGHT" \
  --speed-curvature-loss-weight "$SPEED_CURVATURE_LOSS_WEIGHT" \
  --traj-delta-loss-weight "$TRAJ_DELTA_LOSS_WEIGHT" \
  --traj-curvature-loss-weight "$TRAJ_CURVATURE_LOSS_WEIGHT" \
  --prior-loss-weight "$PRIOR_LOSS_WEIGHT" \
  --stop-loss-weight "$STOP_LOSS_WEIGHT" \
  --stop-state-loss-weight "$STOP_STATE_LOSS_WEIGHT" \
  --stop-reason-loss-weight "$STOP_REASON_LOSS_WEIGHT" \
  --stop-positive-loss-scale "$STOP_POSITIVE_LOSS_SCALE" \
  --stop-negative-loss-scale "$STOP_NEGATIVE_LOSS_SCALE" \
  --stop-loss-after-epoch "$STOP_LOSS_AFTER_EPOCH" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$OUT/train.log"
