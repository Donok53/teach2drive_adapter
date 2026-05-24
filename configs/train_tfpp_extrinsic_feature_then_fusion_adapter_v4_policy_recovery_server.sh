#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Extrinsic-aware Feature+Fusion Adapter v4: policy-recovery training.
#
# This intentionally does not use sideguard or any evaluation-time safety shim.
# The deployable checkpoint remains the same feature-then-fusion adapter loaded
# by scripts/tfpp_feature_then_fusion_adapter_sensor_rig_agent.py.  During
# training only, an auxiliary policy head pushes the adapted fused feature to
# recover canonical TransFuser++ target/speed/stop behavior.

export PY=${PY:-python}
export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}
export BASE_WORK_ROOT=${BASE_WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_feature_then_fusion_adapter_v1"}
export V2_WORK_ROOT=${V2_WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_extrinsic_feature_then_fusion_adapter_v2_conservative"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_extrinsic_feature_then_fusion_adapter_v4_policy_recovery"}
export BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-"$HOME/teach2drive/workspace/teach2drive_bootstrap"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
export TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}
export TFPP_CHECKPOINT=${TFPP_CHECKPOINT:-""}

# Reuse the exact v1 feature/fusion cache preparation path so v4 changes only
# the training objective, not the data export/index/cache pipeline.
export VIEW_ROOT=${VIEW_ROOT:-"$BASE_WORK_ROOT/profile_views"}
export INDEX_DIR=${INDEX_DIR:-"$BASE_WORK_ROOT/indexes"}
export FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR:-"$BASE_WORK_ROOT/feature_fusion_cache"}
export TFPP_VIEW=${TFPP_VIEW:-"$VIEW_ROOT/tfpp_ego"}
export SHIFT_VIEW=${SHIFT_VIEW:-"$VIEW_ROOT/front_triplet_shifted"}
export TFPP_INDEX=${TFPP_INDEX:-"$INDEX_DIR/tfpp_ego_index.npz"}
export SHIFT_INDEX=${SHIFT_INDEX:-"$INDEX_DIR/front_triplet_shifted_index.npz"}
export TFPP_FEATURE_CACHE=${TFPP_FEATURE_CACHE:-"$FEATURE_CACHE_DIR/tfpp_ego_feature_fusion"}
export SHIFT_FEATURE_CACHE=${SHIFT_FEATURE_CACHE:-"$FEATURE_CACHE_DIR/front_triplet_shifted_feature_fusion"}

export OUT=${OUT:-"$WORK_ROOT/train_extrinsic_feature_then_fusion_adapter_v4_policy_recovery"}
export TEACHER_CACHE=${TEACHER_CACHE:-"$WORK_ROOT/cache/tfpp_ego_prior_cache.npz"}
export INIT_CHECKPOINT=${INIT_CHECKPOINT:-"$V2_WORK_ROOT/train_extrinsic_feature_then_fusion_adapter_v2_conservative/best_model.pt"}

export EXPORT_OVERWRITE=${EXPORT_OVERWRITE:-0}
export INDEX_OVERWRITE=${INDEX_OVERWRITE:-0}
export OVERWRITE=${OVERWRITE:-0}
export OVERWRITE_TEACHER_CACHE=${OVERWRITE_TEACHER_CACHE:-0}
export SKIP_INVALID_MOTION=${SKIP_INVALID_MOTION:-1}
export AUGMENTATIONS=${AUGMENTATIONS:-0}
export SEED=${SEED:-41}

export FEATURE_CACHE_BATCH_SIZE=${FEATURE_CACHE_BATCH_SIZE:-32}
export FEATURE_CACHE_WORKERS=${FEATURE_CACHE_WORKERS:-8}
export FEATURE_CACHE_DTYPE=${FEATURE_CACHE_DTYPE:-float16}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
export CACHE_WORKERS=${CACHE_WORKERS:-16}

export EXTRINSIC_AWARE=${EXTRINSIC_AWARE:-1}
export SOURCE_PROFILE=${SOURCE_PROFILE:-front_triplet_shifted}
export EXTRINSIC_HIDDEN_DIM=${EXTRINSIC_HIDDEN_DIM:-64}
export EXTRINSIC_DROPOUT=${EXTRINSIC_DROPOUT:-0.0}
export FREEZE_BASE=${FREEZE_BASE:-0}

export EPOCHS=${EPOCHS:-45}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-10}
export EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.0}
export BATCH_SIZE=${BATCH_SIZE:-128}
export NUM_WORKERS=${NUM_WORKERS:-4}
export LR=${LR:-5e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export VAL_RATIO=${VAL_RATIO:-0.15}
export HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-0}
export BLOCKS=${BLOCKS:-2}
export DROPOUT=${DROPOUT:-0.0}
export DATA_PARALLEL=${DATA_PARALLEL:-1}

# Feature alignment remains present, but the fused feature is now also trained
# against behavior-critical policy supervision from canonical TF++.
export STAGE_LOSS_WEIGHT=${STAGE_LOSS_WEIGHT:-1.0}
export FUSED_LOSS_WEIGHT=${FUSED_LOSS_WEIGHT:-1.0}
export STAGE_FEATURE_LOSS_WEIGHT=${STAGE_FEATURE_LOSS_WEIGHT:-1.0}
export STAGE_COSINE_LOSS_WEIGHT=${STAGE_COSINE_LOSS_WEIGHT:-0.04}
export STAGE_RESIDUAL_LOSS_WEIGHT=${STAGE_RESIDUAL_LOSS_WEIGHT:-0.035}
export STAGE_BASE_CONSISTENCY_LOSS_WEIGHT=${STAGE_BASE_CONSISTENCY_LOSS_WEIGHT:-0.10}
export FUSED_FEATURE_LOSS_WEIGHT=${FUSED_FEATURE_LOSS_WEIGHT:-1.0}
export FUSED_COSINE_LOSS_WEIGHT=${FUSED_COSINE_LOSS_WEIGHT:-0.04}
export FUSED_RESIDUAL_LOSS_WEIGHT=${FUSED_RESIDUAL_LOSS_WEIGHT:-0.045}
export FUSED_BASE_CONSISTENCY_LOSS_WEIGHT=${FUSED_BASE_CONSISTENCY_LOSS_WEIGHT:-0.18}

export POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-0.45}
export POLICY_LOSS_AFTER_EPOCH=${POLICY_LOSS_AFTER_EPOCH:-1}
export POLICY_HIDDEN_DIM=${POLICY_HIDDEN_DIM:-512}
export POLICY_DROPOUT=${POLICY_DROPOUT:-0.02}
export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-0.40}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.03}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.22}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.04}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.04}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.02}

# Teacher blending nudges the adapter toward the original TransFuser++ policy
# while still respecting dataset labels.  This is the part meant to recover
# launch/release timing instead of solving it with sideguard.
export TEACHER_TRAJ_BLEND=${TEACHER_TRAJ_BLEND:-0.75}
export TEACHER_SPEED_TARGET_BLEND=${TEACHER_SPEED_TARGET_BLEND:-0.45}
export TEACHER_STOP_TARGET_BLEND=${TEACHER_STOP_TARGET_BLEND:-0.0}
export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.35}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-1.0}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.04}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-1.0}
export SPEED_FLOOR_TARGET_THRESHOLD=${SPEED_FLOOR_TARGET_THRESHOLD:-2.0}
export STOP_SPEED_CEILING_LOSS_WEIGHT=${STOP_SPEED_CEILING_LOSS_WEIGHT:-0.04}
export STOP_SPEED_CEILING_MPS=${STOP_SPEED_CEILING_MPS:-0.5}
export STOP_SPEED_TARGET_THRESHOLD=${STOP_SPEED_TARGET_THRESHOLD:-0.5}
export LAUNCH_CURRENT_SPEED_THRESHOLD=${LAUNCH_CURRENT_SPEED_THRESHOLD:-0.8}
export LAUNCH_TARGET_SPEED_THRESHOLD=${LAUNCH_TARGET_SPEED_THRESHOLD:-2.0}
export LAUNCH_SAMPLE_WEIGHT=${LAUNCH_SAMPLE_WEIGHT:-1.6}
export LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=${LAUNCH_SPEED_FLOOR_LOSS_WEIGHT:-0.08}
export LAUNCH_SPEED_FLOOR_MPS=${LAUNCH_SPEED_FLOOR_MPS:-2.0}
export RELEASE_TARGET_SPEED_THRESHOLD=${RELEASE_TARGET_SPEED_THRESHOLD:-1.0}
export RELEASE_SAMPLE_WEIGHT=${RELEASE_SAMPLE_WEIGHT:-1.5}
export RELEASE_SPEED_FLOOR_LOSS_WEIGHT=${RELEASE_SPEED_FLOOR_LOSS_WEIGHT:-0.06}
export RELEASE_SPEED_FLOOR_MPS=${RELEASE_SPEED_FLOOR_MPS:-1.5}
export GRAD_CLIP=${GRAD_CLIP:-1.0}

mkdir -p "$WORK_ROOT" "$OUT" "$(dirname "$TEACHER_CACHE")"

echo "=== prepare aligned feature/fusion caches"
(
  export WORK_ROOT="$BASE_WORK_ROOT"
  export OUT="$BASE_WORK_ROOT/train_feature_then_fusion_adapter_v1"
  export TRAIN_ADAPTER=0
  bash configs/train_tfpp_feature_then_fusion_adapter_v1_server.sh
)

CACHE_ARGS=()
if [[ "$DATA_PARALLEL" == "1" || "$DATA_PARALLEL" == "true" || "$DATA_PARALLEL" == "TRUE" ]]; then
  CACHE_ARGS+=(--data-parallel)
fi
if [[ -n "$TFPP_CHECKPOINT" ]]; then
  CACHE_ARGS+=(--checkpoint "$TFPP_CHECKPOINT")
fi

if [[ -f "$TEACHER_CACHE" && "$OVERWRITE" != "1" && "$OVERWRITE_TEACHER_CACHE" != "1" ]]; then
  echo "=== reuse canonical TF++ teacher prior $TEACHER_CACHE"
else
  echo "=== cache canonical TF++ teacher prior"
  rm -f "$TEACHER_CACHE"
  "$PY" -m teach2drive_adapter.cache_transfuserpp_prior \
    --index "$TFPP_INDEX" \
    --output "$TEACHER_CACHE" \
    --garage-root "$GARAGE_ROOT" \
    --team-config "$TEAM_CONFIG" \
    --episode-root-override "$TFPP_VIEW" \
    --cameras front \
    --tfpp-camera front \
    --command-mode target_angle \
    --image-size 640 360 \
    --lidar-size 128 \
    --batch-size "$CACHE_BATCH_SIZE" \
    --num-workers "$CACHE_WORKERS" \
    "${CACHE_ARGS[@]}"
fi

TRAIN_ARGS=()
if [[ "$DATA_PARALLEL" == "1" || "$DATA_PARALLEL" == "true" || "$DATA_PARALLEL" == "TRUE" ]]; then
  TRAIN_ARGS+=(--data-parallel)
fi
if [[ "$EXTRINSIC_AWARE" == "1" || "$EXTRINSIC_AWARE" == "true" || "$EXTRINSIC_AWARE" == "TRUE" ]]; then
  TRAIN_ARGS+=(--extrinsic-aware)
fi
if [[ "$FREEZE_BASE" == "1" || "$FREEZE_BASE" == "true" || "$FREEZE_BASE" == "TRUE" ]]; then
  TRAIN_ARGS+=(--freeze-base)
fi
if [[ -n "$INIT_CHECKPOINT" && -f "$INIT_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
elif [[ -n "$INIT_CHECKPOINT" ]]; then
  echo "=== init checkpoint missing, training from random adapter init: $INIT_CHECKPOINT"
fi

echo "=== train extrinsic feature+fusion adapter v4 policy recovery"
PYTHONUNBUFFERED=1 "$PY" -m teach2drive_adapter.train_transfuserpp_feature_then_fusion_policy_adapter \
  --source-cache "$SHIFT_FEATURE_CACHE" \
  --target-cache "$TFPP_FEATURE_CACHE" \
  --teacher-cache "$TEACHER_CACHE" \
  --index "$SHIFT_INDEX" \
  --out-dir "$OUT" \
  --source-profile "$SOURCE_PROFILE" \
  --extrinsic-hidden-dim "$EXTRINSIC_HIDDEN_DIM" \
  --extrinsic-dropout "$EXTRINSIC_DROPOUT" \
  --epochs "$EPOCHS" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --hidden-channels "$HIDDEN_CHANNELS" \
  --blocks "$BLOCKS" \
  --dropout "$DROPOUT" \
  --stage-loss-weight "$STAGE_LOSS_WEIGHT" \
  --fused-loss-weight "$FUSED_LOSS_WEIGHT" \
  --stage-feature-loss-weight "$STAGE_FEATURE_LOSS_WEIGHT" \
  --stage-cosine-loss-weight "$STAGE_COSINE_LOSS_WEIGHT" \
  --stage-residual-loss-weight "$STAGE_RESIDUAL_LOSS_WEIGHT" \
  --stage-base-consistency-loss-weight "$STAGE_BASE_CONSISTENCY_LOSS_WEIGHT" \
  --fused-feature-loss-weight "$FUSED_FEATURE_LOSS_WEIGHT" \
  --fused-cosine-loss-weight "$FUSED_COSINE_LOSS_WEIGHT" \
  --fused-residual-loss-weight "$FUSED_RESIDUAL_LOSS_WEIGHT" \
  --fused-base-consistency-loss-weight "$FUSED_BASE_CONSISTENCY_LOSS_WEIGHT" \
  --policy-loss-weight "$POLICY_LOSS_WEIGHT" \
  --policy-loss-after-epoch "$POLICY_LOSS_AFTER_EPOCH" \
  --policy-hidden-dim "$POLICY_HIDDEN_DIM" \
  --policy-dropout "$POLICY_DROPOUT" \
  --xy-loss-weight "$XY_LOSS_WEIGHT" \
  --yaw-loss-weight "$YAW_LOSS_WEIGHT" \
  --speed-loss-weight "$SPEED_LOSS_WEIGHT" \
  --stop-loss-weight "$STOP_LOSS_WEIGHT" \
  --stop-state-loss-weight "$STOP_STATE_LOSS_WEIGHT" \
  --stop-reason-loss-weight "$STOP_REASON_LOSS_WEIGHT" \
  --teacher-traj-blend "$TEACHER_TRAJ_BLEND" \
  --teacher-speed-target-blend "$TEACHER_SPEED_TARGET_BLEND" \
  --teacher-stop-target-blend "$TEACHER_STOP_TARGET_BLEND" \
  --moving-sample-weight "$MOVING_SAMPLE_WEIGHT" \
  --stopped-sample-weight "$STOPPED_SAMPLE_WEIGHT" \
  --speed-floor-loss-weight "$SPEED_FLOOR_LOSS_WEIGHT" \
  --speed-floor-mps "$SPEED_FLOOR_MPS" \
  --speed-floor-target-threshold "$SPEED_FLOOR_TARGET_THRESHOLD" \
  --stop-speed-ceiling-loss-weight "$STOP_SPEED_CEILING_LOSS_WEIGHT" \
  --stop-speed-ceiling-mps "$STOP_SPEED_CEILING_MPS" \
  --stop-speed-target-threshold "$STOP_SPEED_TARGET_THRESHOLD" \
  --launch-current-speed-threshold "$LAUNCH_CURRENT_SPEED_THRESHOLD" \
  --launch-target-speed-threshold "$LAUNCH_TARGET_SPEED_THRESHOLD" \
  --launch-sample-weight "$LAUNCH_SAMPLE_WEIGHT" \
  --launch-speed-floor-loss-weight "$LAUNCH_SPEED_FLOOR_LOSS_WEIGHT" \
  --launch-speed-floor-mps "$LAUNCH_SPEED_FLOOR_MPS" \
  --release-target-speed-threshold "$RELEASE_TARGET_SPEED_THRESHOLD" \
  --release-sample-weight "$RELEASE_SAMPLE_WEIGHT" \
  --release-speed-floor-loss-weight "$RELEASE_SPEED_FLOOR_LOSS_WEIGHT" \
  --release-speed-floor-mps "$RELEASE_SPEED_FLOOR_MPS" \
  --grad-clip "$GRAD_CLIP" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$OUT/train.log"
