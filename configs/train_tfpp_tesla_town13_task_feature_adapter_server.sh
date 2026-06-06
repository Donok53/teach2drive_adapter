#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Target-only task-driven feature adapter for the Tesla Town13 dataset.
#
# This script intentionally trains from only the target vehicle/sensor rig:
#
#   front_triplet_shifted sensors -> frozen TransFuser++ with inserted
#   feature-then-fusion adapter -> TransFuser++ waypoint/speed outputs
#   -> expert trajectory/speed/stop losses.
#
# It does not export, cache, or train against tfpp_ego/canonical sensor views.

export PY=${PY:-python}
export SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_town13_paired_tfpp_ego_front_triplet_3h"}
export SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_tesla_town13_front_triplet_target_only_snapshot_complete"}
export SNAPSHOT_COMPLETE_EPISODES=${SNAPSHOT_COMPLETE_EPISODES:-1}
export REFRESH_SNAPSHOT=${REFRESH_SNAPSHOT:-1}
export SNAPSHOT_MIN_FRAMES=${SNAPSHOT_MIN_FRAMES:-1150}
export SNAPSHOT_REQUIRED_PROFILES=${SNAPSHOT_REQUIRED_PROFILES:-"front_triplet_shifted"}

export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_tesla_town13_task_feature_adapter"}
export BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-"$HOME/teach2drive/workspace/teach2drive_bootstrap"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
export TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}
export TFPP_CHECKPOINT=${TFPP_CHECKPOINT:-""}

export PROFILE=${PROFILE:-front_triplet_shifted}
export CAMERAS=${CAMERAS:-left,front,right}
export VIEW_ROOT=${VIEW_ROOT:-"$WORK_ROOT/profile_views"}
export TARGET_VIEW=${TARGET_VIEW:-"$VIEW_ROOT/$PROFILE"}
export INDEX_DIR=${INDEX_DIR:-"$WORK_ROOT/indexes"}
export TARGET_INDEX=${TARGET_INDEX:-"$INDEX_DIR/${PROFILE}_index.npz"}
export OUT=${OUT:-"$WORK_ROOT/train_${PROFILE}_task_feature_adapter"}

export EXPORT_OVERWRITE=${EXPORT_OVERWRITE:-0}
export SKIP_EXPORT=${SKIP_EXPORT:-0}
export INDEX_OVERWRITE=${INDEX_OVERWRITE:-0}
export OVERWRITE=${OVERWRITE:-0}
export SKIP_INVALID_MOTION=${SKIP_INVALID_MOTION:-1}
export AUGMENTATIONS=${AUGMENTATIONS:-0}
export SEED=${SEED:-41}

export IMAGE_WIDTH=${IMAGE_WIDTH:-640}
export IMAGE_HEIGHT=${IMAGE_HEIGHT:-360}
export CAMERA_CROP_SHIFT_X_PX=${CAMERA_CROP_SHIFT_X_PX:-0.0}
export CAMERA_CROP_SHIFT_Y_PX=${CAMERA_CROP_SHIFT_Y_PX:-0.0}
export CAMERA_CROP_SCALE=${CAMERA_CROP_SCALE:-1.0}
export LIDAR_SIZE=${LIDAR_SIZE:-128}
export EPOCHS=${EPOCHS:-20}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-6}
export EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.0}
export SELECTION_METRIC=${SELECTION_METRIC:-loss}
export SELECTION_MODE=${SELECTION_MODE:-min}
export SAVE_EPOCH_CHECKPOINTS=${SAVE_EPOCH_CHECKPOINTS:-0}
export EPOCH_CHECKPOINT_DIR=${EPOCH_CHECKPOINT_DIR:-epoch_checkpoints}
export BATCH_SIZE=${BATCH_SIZE:-8}
export NUM_WORKERS=${NUM_WORKERS:-4}
export LR=${LR:-2e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export DATA_PARALLEL=${DATA_PARALLEL:-0}
export VAL_RATIO=${VAL_RATIO:-0.15}

export EXTRINSIC_AWARE=${EXTRINSIC_AWARE:-1}
export SOURCE_PROFILE=${SOURCE_PROFILE:-front_triplet_shifted}
export EXTRINSIC_HIDDEN_DIM=${EXTRINSIC_HIDDEN_DIM:-64}
export EXTRINSIC_DROPOUT=${EXTRINSIC_DROPOUT:-0.0}
export HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-0}
export BLOCKS=${BLOCKS:-2}
export DROPOUT=${DROPOUT:-0.0}
export STAGE_ADAPTER_LAYERS=${STAGE_ADAPTER_LAYERS:-all}
export STAGE_ADAPTER_MODALITIES=${STAGE_ADAPTER_MODALITIES:-all}
export FUSION_ADAPTER_ENABLED=${FUSION_ADAPTER_ENABLED:-1}
export STAGE_FEATURE_ADAPTER_BLEND=${STAGE_FEATURE_ADAPTER_BLEND:-1.0}
export FUSION_ADAPTER_BLEND=${FUSION_ADAPTER_BLEND:-1.0}
export INIT_CHECKPOINT=${INIT_CHECKPOINT:-""}
export LORA_RANK=${LORA_RANK:-0}
export LORA_ALPHA=${LORA_ALPHA:-16.0}
export LORA_DROPOUT=${LORA_DROPOUT:-0.0}
export LORA_INCLUDE=${LORA_INCLUDE:-"^join\\.,^checkpoint_decoder\\.(encoder|decoder)\\.,^target_speed_network\\."}
export LORA_EXCLUDE=${LORA_EXCLUDE:-""}
export UNFREEZE_INCLUDE=${UNFREEZE_INCLUDE:-""}
export UNFREEZE_EXCLUDE=${UNFREEZE_EXCLUDE:-""}
export UNFREEZE_LR=${UNFREEZE_LR:-0.0}
export UNFREEZE_WEIGHT_DECAY=${UNFREEZE_WEIGHT_DECAY:-1e-5}
export LIDAR_CANONICAL_SHIFT_X_M=${LIDAR_CANONICAL_SHIFT_X_M:-0.0}
export LIDAR_CANONICAL_SHIFT_Y_M=${LIDAR_CANONICAL_SHIFT_Y_M:-0.0}
export LIDAR_PIXELS_PER_METER=${LIDAR_PIXELS_PER_METER:-4.0}

export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-0.55}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.03}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.80}
export TRAJ_SMOOTH_LOSS_WEIGHT=${TRAJ_SMOOTH_LOSS_WEIGHT:-0.0}
export SPEED_SMOOTH_LOSS_WEIGHT=${SPEED_SMOOTH_LOSS_WEIGHT:-0.0}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.08}
export FEATURE_DRIFT_LOSS_WEIGHT=${FEATURE_DRIFT_LOSS_WEIGHT:-0.10}
export OUTPUT_PRIOR_XY_LOSS_WEIGHT=${OUTPUT_PRIOR_XY_LOSS_WEIGHT:-0.0}
export OUTPUT_PRIOR_SPEED_LOSS_WEIGHT=${OUTPUT_PRIOR_SPEED_LOSS_WEIGHT:-0.0}
export AUX_HIDDEN_DIM=${AUX_HIDDEN_DIM:-256}
export CONTROL_LOSS_WEIGHT=${CONTROL_LOSS_WEIGHT:-0.0}
export OUTPUT_RESIDUAL=${OUTPUT_RESIDUAL:-0}
export OUTPUT_RESIDUAL_HIDDEN_DIM=${OUTPUT_RESIDUAL_HIDDEN_DIM:-256}
export OUTPUT_RESIDUAL_CHECKPOINT_SCALE=${OUTPUT_RESIDUAL_CHECKPOINT_SCALE:-0.75}
export OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE=${OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE:-1.5}
export OUTPUT_RESIDUAL_GATE_BIAS=${OUTPUT_RESIDUAL_GATE_BIAS:--2.0}
export OUTPUT_RESIDUAL_DROPOUT=${OUTPUT_RESIDUAL_DROPOUT:-0.0}
export STOP_STATE_AUX_LOSS_WEIGHT=${STOP_STATE_AUX_LOSS_WEIGHT:-0.0}
export STOP_REASON_AUX_LOSS_WEIGHT=${STOP_REASON_AUX_LOSS_WEIGHT:-0.0}

export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.15}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-1.0}
export HAZARD_STOP_REASONS=${HAZARD_STOP_REASONS:-traffic_light,stop_sign,front_vehicle,junction_yield}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-2.0}

export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.03}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-0.8}
export SPEED_FLOOR_TARGET_THRESHOLD=${SPEED_FLOOR_TARGET_THRESHOLD:-2.0}
export STOP_SPEED_CEILING_LOSS_WEIGHT=${STOP_SPEED_CEILING_LOSS_WEIGHT:-0.35}
export STOP_SPEED_CEILING_MPS=${STOP_SPEED_CEILING_MPS:-0.5}
export STOP_SPEED_TARGET_THRESHOLD=${STOP_SPEED_TARGET_THRESHOLD:-0.5}
export STOP_PROGRESS_CEILING_M=${STOP_PROGRESS_CEILING_M:-1.0}
export GO_PROGRESS_RATIO=${GO_PROGRESS_RATIO:-0.5}
export LAUNCH_SAMPLE_WEIGHT=${LAUNCH_SAMPLE_WEIGHT:-1.4}
export LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=${LAUNCH_SPEED_FLOOR_LOSS_WEIGHT:-0.04}
export LAUNCH_SPEED_FLOOR_MPS=${LAUNCH_SPEED_FLOOR_MPS:-1.2}
export RELEASE_SAMPLE_WEIGHT=${RELEASE_SAMPLE_WEIGHT:-1.3}
export RELEASE_SPEED_FLOOR_LOSS_WEIGHT=${RELEASE_SPEED_FLOOR_LOSS_WEIGHT:-0.04}
export RELEASE_SPEED_FLOOR_MPS=${RELEASE_SPEED_FLOOR_MPS:-1.2}
export CONTROLLER_STEER_CLOSE_THRESHOLD=${CONTROLLER_STEER_CLOSE_THRESHOLD:-0.15}
export CONTROLLER_THROTTLE_CLOSE_THRESHOLD=${CONTROLLER_THROTTLE_CLOSE_THRESHOLD:-0.20}
export CONTROLLER_GO_THROTTLE_THRESHOLD=${CONTROLLER_GO_THROTTLE_THRESHOLD:-0.05}
export GRAD_CLIP=${GRAD_CLIP:-1.0}

mkdir -p "$WORK_ROOT" "$VIEW_ROOT" "$INDEX_DIR"

if [[ "$SNAPSHOT_COMPLETE_EPISODES" == "1" ]]; then
  echo "=== snapshot complete target-rig episodes from $SOURCE_DATA_ROOT -> $SNAPSHOT_ROOT"
  SOURCE_DATA_ROOT="$SOURCE_DATA_ROOT" \
  SNAPSHOT_ROOT="$SNAPSHOT_ROOT" \
  REFRESH_SNAPSHOT="$REFRESH_SNAPSHOT" \
  SNAPSHOT_MIN_FRAMES="$SNAPSHOT_MIN_FRAMES" \
  SNAPSHOT_REQUIRED_PROFILES="$SNAPSHOT_REQUIRED_PROFILES" \
  "$PY" - <<'PY'
import json
import os
import shutil
from pathlib import Path

src = Path(os.environ["SOURCE_DATA_ROOT"]).expanduser()
dst = Path(os.environ["SNAPSHOT_ROOT"]).expanduser()
min_frames = int(os.environ.get("SNAPSHOT_MIN_FRAMES", "1150"))
profiles = [p.strip() for p in os.environ.get("SNAPSHOT_REQUIRED_PROFILES", "").split(",") if p.strip()]

if os.environ.get("REFRESH_SNAPSHOT", "1") == "1" and dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)

accepted = []
rejected = []
for ep in sorted(src.glob("episode_*")):
    if not ep.is_dir():
        continue
    frames = ep / "frames.jsonl"
    if not frames.exists():
        rejected.append((ep.name, "missing frames.jsonl"))
        continue
    missing_layouts = [
        profile
        for profile in profiles
        if not (ep / "rigs" / profile / "sensor_layout.json").exists()
    ]
    if missing_layouts:
        rejected.append((ep.name, "missing layouts=" + ",".join(missing_layouts)))
        continue
    try:
        frame_count = sum(1 for _ in frames.open("r", encoding="utf-8"))
    except OSError as exc:
        rejected.append((ep.name, f"frames read failed: {exc}"))
        continue
    if frame_count < min_frames:
        rejected.append((ep.name, f"frames={frame_count} < {min_frames}"))
        continue
    target = dst / ep.name
    if not target.exists():
        target.symlink_to(ep, target_is_directory=True)
    accepted.append((ep.name, frame_count))

print(json.dumps({
    "source": str(src),
    "snapshot": str(dst),
    "required_profiles": profiles,
    "accepted": len(accepted),
    "accepted_tail": accepted[-8:],
    "rejected": len(rejected),
    "rejected_tail": rejected[-8:],
}, indent=2, ensure_ascii=False))
if not accepted:
    raise SystemExit("no complete target-rig episodes available for training")
PY
  export DATA_ROOT="$SNAPSHOT_ROOT"
else
  export DATA_ROOT=${DATA_ROOT:-"$SOURCE_DATA_ROOT"}
fi

EXPORT_ARGS=()
if [[ "$EXPORT_OVERWRITE" == "1" ]]; then
  EXPORT_ARGS+=(--overwrite)
fi
if [[ "$SKIP_INVALID_MOTION" == "1" || "$SKIP_INVALID_MOTION" == "true" || "$SKIP_INVALID_MOTION" == "TRUE" ]]; then
  EXPORT_ARGS+=(--skip-invalid-motion)
fi

if [[ "$SKIP_EXPORT" == "1" || "$SKIP_EXPORT" == "true" || "$SKIP_EXPORT" == "TRUE" ]]; then
  if [[ ! -d "$TARGET_VIEW" ]]; then
    echo "SKIP_EXPORT requested but TARGET_VIEW does not exist: $TARGET_VIEW" >&2
    exit 1
  fi
  echo "=== reuse profile view $TARGET_VIEW"
else
  echo "=== export $PROFILE profile view"
  "$PY" -m teach2drive_adapter.export_paired_profile_view \
    --input-root "$DATA_ROOT" \
    --output-root "$TARGET_VIEW" \
    --profile "$PROFILE" \
    --require-cameras "$CAMERAS" \
    "${EXPORT_ARGS[@]}"
fi

if [[ -f "$TARGET_INDEX" && "$INDEX_OVERWRITE" != "1" ]]; then
  echo "=== reuse index $TARGET_INDEX"
else
  echo "=== build index $TARGET_INDEX cameras=$CAMERAS"
  (
    cd "$BOOTSTRAP_ROOT"
    PYTHONPATH="$BOOTSTRAP_ROOT:${PYTHONPATH:-}" "$PY" -m teach2drive.token_dataset \
      --input-root "$TARGET_VIEW" \
      --output "$TARGET_INDEX" \
      --cameras "$CAMERAS" \
      --augmentations "$AUGMENTATIONS" \
      --pseudo-label-name "__missing_pseudo_labels__.jsonl" \
      --seed "$SEED"
  )
fi

if [[ -f "$OUT/best_model.pt" && "$OVERWRITE" != "1" ]]; then
  echo "=== reuse trained task feature adapter $OUT/best_model.pt"
  exit 0
fi
if [[ "$OVERWRITE" == "1" ]]; then
  rm -rf "$OUT"
fi
mkdir -p "$OUT"

TRAIN_ARGS=()
if [[ "$EXTRINSIC_AWARE" == "1" || "$EXTRINSIC_AWARE" == "true" || "$EXTRINSIC_AWARE" == "TRUE" ]]; then
  TRAIN_ARGS+=(--extrinsic-aware)
fi
if [[ -n "$TFPP_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--checkpoint "$TFPP_CHECKPOINT")
fi
if [[ -n "$INIT_CHECKPOINT" ]]; then
  TRAIN_ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
fi
if [[ "$DATA_PARALLEL" == "1" || "$DATA_PARALLEL" == "true" || "$DATA_PARALLEL" == "TRUE" ]]; then
  TRAIN_ARGS+=(--data-parallel)
fi
if [[ "$SAVE_EPOCH_CHECKPOINTS" == "1" || "$SAVE_EPOCH_CHECKPOINTS" == "true" || "$SAVE_EPOCH_CHECKPOINTS" == "TRUE" ]]; then
  TRAIN_ARGS+=(--save-epoch-checkpoints)
fi
if [[ "$FUSION_ADAPTER_ENABLED" == "0" || "$FUSION_ADAPTER_ENABLED" == "false" || "$FUSION_ADAPTER_ENABLED" == "FALSE" ]]; then
  TRAIN_ARGS+=(--disable-fusion-adapter)
fi
if [[ "$OUTPUT_RESIDUAL" == "1" || "$OUTPUT_RESIDUAL" == "true" || "$OUTPUT_RESIDUAL" == "TRUE" ]]; then
  TRAIN_ARGS+=(--output-residual)
fi

echo "=== train target-only task feature adapter"
PYTHONUNBUFFERED=1 "$PY" -m teach2drive_adapter.train_transfuserpp_task_feature_adapter \
  --index "$TARGET_INDEX" \
  --episode-root-override "$TARGET_VIEW" \
  --out-dir "$OUT" \
  --garage-root "$GARAGE_ROOT" \
  --team-config "$TEAM_CONFIG" \
  --cameras "$CAMERAS" \
  --tfpp-camera front \
  --command-mode target_angle \
  --image-size "$IMAGE_WIDTH" "$IMAGE_HEIGHT" \
  --camera-crop-shift-x-px "$CAMERA_CROP_SHIFT_X_PX" \
  --camera-crop-shift-y-px "$CAMERA_CROP_SHIFT_Y_PX" \
  --camera-crop-scale "$CAMERA_CROP_SCALE" \
  --lidar-size "$LIDAR_SIZE" \
  --source-profile "$SOURCE_PROFILE" \
  --extrinsic-hidden-dim "$EXTRINSIC_HIDDEN_DIM" \
  --extrinsic-dropout "$EXTRINSIC_DROPOUT" \
  --hidden-channels "$HIDDEN_CHANNELS" \
  --blocks "$BLOCKS" \
  --dropout "$DROPOUT" \
  --stage-adapter-layers "$STAGE_ADAPTER_LAYERS" \
  --stage-adapter-modalities "$STAGE_ADAPTER_MODALITIES" \
  --stage-feature-adapter-blend "$STAGE_FEATURE_ADAPTER_BLEND" \
  --fusion-adapter-blend "$FUSION_ADAPTER_BLEND" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --lora-include "$LORA_INCLUDE" \
  --lora-exclude "$LORA_EXCLUDE" \
  --unfreeze-include "$UNFREEZE_INCLUDE" \
  --unfreeze-exclude "$UNFREEZE_EXCLUDE" \
  --unfreeze-lr "$UNFREEZE_LR" \
  --unfreeze-weight-decay "$UNFREEZE_WEIGHT_DECAY" \
  --lidar-canonical-shift-x-m "$LIDAR_CANONICAL_SHIFT_X_M" \
  --lidar-canonical-shift-y-m "$LIDAR_CANONICAL_SHIFT_Y_M" \
  --lidar-pixels-per-meter "$LIDAR_PIXELS_PER_METER" \
  --epochs "$EPOCHS" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
  --selection-metric "$SELECTION_METRIC" \
  --selection-mode "$SELECTION_MODE" \
  --epoch-checkpoint-dir "$EPOCH_CHECKPOINT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --val-ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --xy-loss-weight "$XY_LOSS_WEIGHT" \
  --yaw-loss-weight "$YAW_LOSS_WEIGHT" \
  --speed-loss-weight "$SPEED_LOSS_WEIGHT" \
  --traj-smooth-loss-weight "$TRAJ_SMOOTH_LOSS_WEIGHT" \
  --speed-smooth-loss-weight "$SPEED_SMOOTH_LOSS_WEIGHT" \
  --stop-loss-weight "$STOP_LOSS_WEIGHT" \
  --feature-drift-loss-weight "$FEATURE_DRIFT_LOSS_WEIGHT" \
  --output-prior-xy-loss-weight "$OUTPUT_PRIOR_XY_LOSS_WEIGHT" \
  --output-prior-speed-loss-weight "$OUTPUT_PRIOR_SPEED_LOSS_WEIGHT" \
  --aux-hidden-dim "$AUX_HIDDEN_DIM" \
  --control-loss-weight "$CONTROL_LOSS_WEIGHT" \
  --output-residual-hidden-dim "$OUTPUT_RESIDUAL_HIDDEN_DIM" \
  --output-residual-checkpoint-scale "$OUTPUT_RESIDUAL_CHECKPOINT_SCALE" \
  --output-residual-speed-logit-scale "$OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE" \
  --output-residual-gate-bias "$OUTPUT_RESIDUAL_GATE_BIAS" \
  --output-residual-dropout "$OUTPUT_RESIDUAL_DROPOUT" \
  --stop-state-aux-loss-weight "$STOP_STATE_AUX_LOSS_WEIGHT" \
  --stop-reason-aux-loss-weight "$STOP_REASON_AUX_LOSS_WEIGHT" \
  --moving-sample-weight "$MOVING_SAMPLE_WEIGHT" \
  --stopped-sample-weight "$STOPPED_SAMPLE_WEIGHT" \
  --hazard-stop-reasons "$HAZARD_STOP_REASONS" \
  --hazard-sample-weight "$HAZARD_SAMPLE_WEIGHT" \
  --speed-floor-loss-weight "$SPEED_FLOOR_LOSS_WEIGHT" \
  --speed-floor-mps "$SPEED_FLOOR_MPS" \
  --speed-floor-target-threshold "$SPEED_FLOOR_TARGET_THRESHOLD" \
  --stop-speed-ceiling-loss-weight "$STOP_SPEED_CEILING_LOSS_WEIGHT" \
  --stop-speed-ceiling-mps "$STOP_SPEED_CEILING_MPS" \
  --stop-speed-target-threshold "$STOP_SPEED_TARGET_THRESHOLD" \
  --stop-progress-ceiling-m "$STOP_PROGRESS_CEILING_M" \
  --go-progress-ratio "$GO_PROGRESS_RATIO" \
  --launch-sample-weight "$LAUNCH_SAMPLE_WEIGHT" \
  --launch-speed-floor-loss-weight "$LAUNCH_SPEED_FLOOR_LOSS_WEIGHT" \
  --launch-speed-floor-mps "$LAUNCH_SPEED_FLOOR_MPS" \
  --release-sample-weight "$RELEASE_SAMPLE_WEIGHT" \
  --release-speed-floor-loss-weight "$RELEASE_SPEED_FLOOR_LOSS_WEIGHT" \
  --release-speed-floor-mps "$RELEASE_SPEED_FLOOR_MPS" \
  --controller-steer-close-threshold "$CONTROLLER_STEER_CLOSE_THRESHOLD" \
  --controller-throttle-close-threshold "$CONTROLLER_THROTTLE_CLOSE_THRESHOLD" \
  --controller-go-throttle-threshold "$CONTROLLER_GO_THROTTLE_THRESHOLD" \
  --grad-clip "$GRAD_CLIP" \
  "${TRAIN_ARGS[@]}" \
  2>&1 | tee "$OUT/train.log"
