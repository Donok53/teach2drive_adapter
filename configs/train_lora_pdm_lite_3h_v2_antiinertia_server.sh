#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ==============================================================================
# v2 = v1 LoRA adaptation + ANTI-INERTIA measures (no extra data).
# Targets the residual "won't launch / min-speed stop-go" (causal-confusion)
# seen in the v1 closed-loop eval. Four levers:
#   (1) ego-speed input dropout/noise  -> T2D_EGO_SPEED_* (transfuserpp_bridge)
#       stops the policy using its own speed as a "if slow -> stay stopped" cue
#   (2) boost launch / speed-floor losses, cut stop-hold losses
#   (3) oversample stop->go transitions via LAUNCH/RELEASE sample weights
#   (4) (eval-side) min-speed floor controller -- applied at eval, not here
# Reuses the already-built t2d 3h token index (no re-conversion / re-index).
# ==============================================================================

PY=${PY:-python}
REPO_ROOT="$PWD"
BASE_CONFIG=${BASE_CONFIG:-configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh}
BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-$HOME/teach2drive/workspace/teach2drive_bootstrap}

SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT:-$HOME/dataset/byeongjae/datasets/t2d_pdm_lite_front_triplet_shifted_3h}
PREP_ROOT=${PREP_ROOT:-$HOME/dataset/byeongjae/runs/lora_pdm_lite_3h_prepared}
RUN_ROOT=${RUN_ROOT:-$HOME/dataset/byeongjae/runs/lora_pdm_lite_3h_v2_antiinertia}
LOG_DIR=${LOG_DIR:-$HOME/teach2drive/logs}
PROFILE=${PROFILE:-front_triplet_shifted}
CAMERAS=${CAMERAS:-left,front,right}
# reuse the v1-built token index (same 3h data) unless overridden
TARGET_INDEX=${TARGET_INDEX:-$PREP_ROOT/indexes/${PROFILE}_index.npz}
TARGET_VIEW=${TARGET_VIEW:-$SOURCE_DATA_ROOT}
GPU=${GPU:-0}

mkdir -p "$PREP_ROOT/indexes" "$RUN_ROOT" "$LOG_DIR"

# --- 1) token index (reuse if present) ------------------------------------------
if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  if [[ -f "$TARGET_INDEX" && "${INDEX_OVERWRITE:-0}" != "1" ]]; then
    echo "=== reuse index $TARGET_INDEX"
  else
    echo "=== build token index $TARGET_INDEX cameras=$CAMERAS"
    ( cd "$BOOTSTRAP_ROOT"
      PYTHONPATH="$BOOTSTRAP_ROOT:$REPO_ROOT:${PYTHONPATH:-}" "$PY" -m teach2drive.token_dataset \
        --input-root "$SOURCE_DATA_ROOT" \
        --output "$TARGET_INDEX" \
        --cameras "$CAMERAS" \
        --augmentations 0 \
        --pseudo-label-name "__missing_pseudo_labels__.jsonl" \
        --seed 91 )
  fi
fi

# --- item (1): ego-speed input augmentation (TRAINING-ONLY, read at import) ------
export T2D_EGO_SPEED_NOISE_STD=${T2D_EGO_SPEED_NOISE_STD:-1.0}   # m/s gaussian noise
export T2D_EGO_SPEED_DROPOUT_P=${T2D_EGO_SPEED_DROPOUT_P:-0.30}  # per-sample zero-out prob

# --- 2) LoRA training with anti-inertia loss/sampling ---------------------------
echo "=== v2 LoRA training -> $RUN_ROOT/train_${PROFILE}_lora"
echo "=== ego_speed_noise=$T2D_EGO_SPEED_NOISE_STD dropout=$T2D_EGO_SPEED_DROPOUT_P"
env \
  CUDA_VISIBLE_DEVICES="$GPU" PY="$PY" PYTHONUNBUFFERED=1 \
  T2D_EGO_SPEED_NOISE_STD="$T2D_EGO_SPEED_NOISE_STD" T2D_EGO_SPEED_DROPOUT_P="$T2D_EGO_SPEED_DROPOUT_P" \
  SNAPSHOT_COMPLETE_EPISODES=0 REFRESH_SNAPSHOT=0 SKIP_EXPORT=1 INDEX_OVERWRITE=0 \
  TARGET_VIEW="$TARGET_VIEW" TARGET_INDEX="$TARGET_INDEX" \
  WORK_ROOT="$RUN_ROOT/work" OUT="$RUN_ROOT/train_${PROFILE}_lora" OVERWRITE=1 \
  SAVE_EPOCH_CHECKPOINTS=1 EPOCH_CHECKPOINT_DIR=epoch_checkpoints \
  BATCH_SIZE=${BATCH_SIZE:-24} NUM_WORKERS=${NUM_WORKERS:-4} \
  EPOCHS=${EPOCHS:-20} EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-8} EARLY_STOP_MIN_DELTA=0.0 \
  SELECTION_METRIC=${SELECTION_METRIC:-loss} SELECTION_MODE=min \
  LR=${LR:-1.0e-5} WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4} \
  LORA_RANK=${LORA_RANK:-8} LORA_ALPHA=${LORA_ALPHA:-16} LORA_DROPOUT=${LORA_DROPOUT:-0.02} \
  LORA_INCLUDE='^join\.,^checkpoint_decoder\.(encoder|decoder)\.,^target_speed_network\.' \
  EXTRINSIC_AWARE=1 SOURCE_PROFILE="$PROFILE" \
  FUSION_ADAPTER_ENABLED=1 STAGE_ADAPTER_MODALITIES=all \
  FEATURE_DRIFT_LOSS_WEIGHT=0.12 \
  `# (2) cut stop-hold pressure` \
  STOP_LOSS_WEIGHT=0.05 STOP_SPEED_CEILING_LOSS_WEIGHT=0.20 \
  STOPPED_SAMPLE_WEIGHT=0.7 HAZARD_SAMPLE_WEIGHT=1.5 \
  `# (3) oversample stop->go transitions + reward motion` \
  MOVING_SAMPLE_WEIGHT=1.4 LAUNCH_SAMPLE_WEIGHT=4.0 RELEASE_SAMPLE_WEIGHT=3.5 \
  `# (2) strengthen launch / speed-floor` \
  SPEED_FLOOR_LOSS_WEIGHT=0.18 LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=0.30 RELEASE_SPEED_FLOOR_LOSS_WEIGHT=0.30 \
  STOP_STATE_AUX_LOSS_WEIGHT=0.15 STOP_REASON_AUX_LOSS_WEIGHT=0.10 \
  TRAJ_SMOOTH_LOSS_WEIGHT=0.030 SPEED_SMOOTH_LOSS_WEIGHT=0.020 CONTROL_LOSS_WEIGHT=0.90 \
  SEED=${SEED:-91} \
  bash "$BASE_CONFIG"

echo "=== done. checkpoint dir: $RUN_ROOT/train_${PROFILE}_lora"
