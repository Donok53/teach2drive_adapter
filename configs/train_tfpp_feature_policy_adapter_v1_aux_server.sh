#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Feature + Policy Adapter v1.
#
# This keeps the same deployable module as the fused-feature adapter, but adds
# an auxiliary policy head during training. The auxiliary head is discarded at
# evaluation time; best_model.pt still stores model_state as the feature adapter
# alone so scripts/tfpp_feature_adapter_sensor_rig_agent.py can load it.
#
# Objective:
#   1) adapted fused feature ~= canonical TF++ fused feature
#   2) adapted fused feature should also predict expert trajectory/speed/stop
#
# The point is to avoid a feature adapter that is numerically aligned but loses
# behavior-critical information for turning, stopping, and release decisions.

PY=${PY:-python}
WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_feature_policy_adapter_v1_aux"}
OUT=${OUT:-"$WORK_ROOT/train_feature_policy_adapter_v1_aux"}

export WORK_ROOT
export TRAIN_ADAPTER=0

# Reuse the exact FA/Fusion cache preparation path so the comparison only
# changes the training objective, not the source data or preprocessing.
bash configs/train_tfpp_feature_adapter_v1_fused_server.sh

SHIFT_FEATURE_CACHE=${SHIFT_FEATURE_CACHE:-"$WORK_ROOT/feature_cache/front_triplet_shifted_fused"}
TFPP_FEATURE_CACHE=${TFPP_FEATURE_CACHE:-"$WORK_ROOT/feature_cache/tfpp_ego_fused"}
SHIFT_INDEX=${SHIFT_INDEX:-"$WORK_ROOT/indexes/front_triplet_shifted_index.npz"}

mkdir -p "$OUT"

TRAIN_ARGS=()
if [[ "${DATA_PARALLEL:-0}" == "1" || "${DATA_PARALLEL:-0}" == "true" || "${DATA_PARALLEL:-0}" == "TRUE" ]]; then
  TRAIN_ARGS+=(--data-parallel)
fi

TEACHER_CACHE_ARG=()
if [[ -n "${TEACHER_CACHE:-}" ]]; then
  TEACHER_CACHE_ARG+=(--teacher-cache "$TEACHER_CACHE")
fi

echo "=== train feature + auxiliary policy adapter v1"
PYTHONUNBUFFERED=1 "$PY" -m teach2drive_adapter.train_transfuserpp_fused_feature_policy_adapter \
  --source-cache "$SHIFT_FEATURE_CACHE" \
  --target-cache "$TFPP_FEATURE_CACHE" \
  --index "$SHIFT_INDEX" \
  --out-dir "$OUT" \
  --epochs "${EPOCHS:-60}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE:-12}" \
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA:-0.0}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-2e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --val-ratio "${VAL_RATIO:-0.15}" \
  --seed "${SEED:-41}" \
  --hidden-channels "${HIDDEN_CHANNELS:-0}" \
  --blocks "${BLOCKS:-3}" \
  --dropout "${DROPOUT:-0.02}" \
  --hidden-dim "${AUX_HIDDEN_DIM:-512}" \
  --feature-loss-weight "${FEATURE_LOSS_WEIGHT:-1.0}" \
  --cosine-loss-weight "${COSINE_LOSS_WEIGHT:-0.08}" \
  --residual-loss-weight "${RESIDUAL_LOSS_WEIGHT:-0.01}" \
  --xy-loss-weight "${XY_LOSS_WEIGHT:-0.35}" \
  --yaw-loss-weight "${YAW_LOSS_WEIGHT:-0.02}" \
  --speed-loss-weight "${SPEED_LOSS_WEIGHT:-0.15}" \
  --stop-loss-weight "${STOP_LOSS_WEIGHT:-0.02}" \
  --stop-state-loss-weight "${STOP_STATE_LOSS_WEIGHT:-0.02}" \
  --stop-reason-loss-weight "${STOP_REASON_LOSS_WEIGHT:-0.01}" \
  --teacher-traj-blend "${TEACHER_TRAJ_BLEND:-0.0}" \
  --teacher-speed-target-blend "${TEACHER_SPEED_TARGET_BLEND:-0.0}" \
  --teacher-stop-target-blend "${TEACHER_STOP_TARGET_BLEND:-0.0}" \
  --moving-sample-weight "${MOVING_SAMPLE_WEIGHT:-1.0}" \
  --stopped-sample-weight "${STOPPED_SAMPLE_WEIGHT:-1.0}" \
  --speed-floor-loss-weight "${SPEED_FLOOR_LOSS_WEIGHT:-0.05}" \
  --stop-speed-ceiling-loss-weight "${STOP_SPEED_CEILING_LOSS_WEIGHT:-0.05}" \
  --launch-sample-weight "${LAUNCH_SAMPLE_WEIGHT:-1.5}" \
  --launch-speed-floor-loss-weight "${LAUNCH_SPEED_FLOOR_LOSS_WEIGHT:-0.05}" \
  --release-sample-weight "${RELEASE_SAMPLE_WEIGHT:-1.5}" \
  --release-speed-floor-loss-weight "${RELEASE_SPEED_FLOOR_LOSS_WEIGHT:-0.05}" \
  "${TEACHER_CACHE_ARG[@]}" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$OUT/train.log"
