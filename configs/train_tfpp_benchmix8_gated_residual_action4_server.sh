#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

# Four shifted-sensor-only gated residual action runs.
#
# These keep the TransFuser++ front-end/backbone behavior mostly frozen and
# learn bounded residuals on the controller-facing checkpoint trajectory and
# target-speed logits.  The residual is trained only against expert labels; the
# shifted TF++ output is used as an input prior, not as a target.

PY="${PY:-/home/jovyan_venv/.venv/torch2.1.2-py3.10-cuda11.8/bin/python}"
BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT:-$HOME/teach2drive/workspace/teach2drive_bootstrap}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-$HOME/dataset/byeongjae/datasets/t2d_tesla_benchmix8_front_triplet_target_3h}"
PREP_ROOT="${PREP_ROOT:-$HOME/dataset/byeongjae/runs/tfpp_tesla_benchmix8_front_triplet_target_3h_prepared}"
RUN_ROOT="${RUN_ROOT:-$HOME/dataset/byeongjae/runs/benchmix8_gated_residual_action}"
LOG_DIR="${LOG_DIR:-$HOME/teach2drive/logs/benchmix8_gated_residual_action}"
PROFILE="${PROFILE:-front_triplet_shifted}"
CAMERAS="${CAMERAS:-left,front,right}"
TARGET_VIEW="${TARGET_VIEW:-$PREP_ROOT/profile_views/$PROFILE}"
TARGET_INDEX="${TARGET_INDEX:-$PREP_ROOT/indexes/${PROFILE}_index.npz}"
EXPORT_OVERWRITE="${EXPORT_OVERWRITE:-0}"
INDEX_OVERWRITE="${INDEX_OVERWRITE:-0}"
SKIP_PREP="${SKIP_PREP:-1}"

mkdir -p "$PREP_ROOT/profile_views" "$PREP_ROOT/indexes" "$RUN_ROOT" "$LOG_DIR"

if [[ "$SKIP_PREP" != "1" ]]; then
  echo "=== export $PROFILE profile view from $SOURCE_DATA_ROOT"
  export_args=()
  if [[ "$EXPORT_OVERWRITE" == "1" ]]; then
    export_args+=(--overwrite)
  fi
  "$PY" -m teach2drive_adapter.export_paired_profile_view \
    --input-root "$SOURCE_DATA_ROOT" \
    --output-root "$TARGET_VIEW" \
    --profile "$PROFILE" \
    --require-cameras "$CAMERAS" \
    --skip-invalid-motion \
    "${export_args[@]}"

  if [[ -f "$TARGET_INDEX" && "$INDEX_OVERWRITE" != "1" ]]; then
    echo "=== reuse index $TARGET_INDEX"
  else
    echo "=== build index $TARGET_INDEX cameras=$CAMERAS"
    (
      cd "$BOOTSTRAP_ROOT"
      PYTHONPATH="$BOOTSTRAP_ROOT:$REPO_ROOT:${PYTHONPATH:-}" "$PY" -m teach2drive.token_dataset \
        --input-root "$TARGET_VIEW" \
        --output "$TARGET_INDEX" \
        --cameras "$CAMERAS" \
        --augmentations 0 \
        --pseudo-label-name "__missing_pseudo_labels__.jsonl" \
        --seed 71
    )
  fi
else
  if [[ ! -d "$TARGET_VIEW" || ! -f "$TARGET_INDEX" ]]; then
    echo "SKIP_PREP=1 but prepared view/index is missing. Re-run with SKIP_PREP=0." >&2
    echo "TARGET_VIEW=$TARGET_VIEW" >&2
    echo "TARGET_INDEX=$TARGET_INDEX" >&2
    exit 1
  fi
fi

echo "=== launch four benchmix8 gated residual action trainings"
echo "TARGET_VIEW=$TARGET_VIEW"
echo "TARGET_INDEX=$TARGET_INDEX"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_DIR=$LOG_DIR"

launch() {
  local name="$1"
  local gpu="$2"
  shift 2
  local session="t2d_${name}"
  local out="$RUN_ROOT/$name"
  local log="$LOG_DIR/$name.log"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session already exists: $session"
    return 0
  fi

  echo "launch $name gpu=$gpu out=$out log=$log"
  local cmd=()
  cmd+=(
    env
    "CUDA_VISIBLE_DEVICES=$gpu"
    "PY=$PY"
    PYTHONUNBUFFERED=1
    SNAPSHOT_COMPLETE_EPISODES=0
    REFRESH_SNAPSHOT=0
    SKIP_EXPORT=1
    INDEX_OVERWRITE=0
    "TARGET_VIEW=$TARGET_VIEW"
    "TARGET_INDEX=$TARGET_INDEX"
    "WORK_ROOT=$RUN_ROOT/${name}_work"
    "OUT=$out"
    OVERWRITE=1
    SAVE_EPOCH_CHECKPOINTS=1
    EPOCH_CHECKPOINT_DIR=epoch_checkpoints
    BATCH_SIZE=24
    NUM_WORKERS=4
    EPOCHS=20
    EARLY_STOP_PATIENCE=8
    EARLY_STOP_MIN_DELTA=0.0
    OUTPUT_PRIOR_XY_LOSS_WEIGHT=0.0
    OUTPUT_PRIOR_SPEED_LOSS_WEIGHT=0.0
    OUTPUT_RESIDUAL=1
    STAGE_ADAPTER_MODALITIES=none
    FUSION_ADAPTER_ENABLED=0
    FEATURE_DRIFT_LOSS_WEIGHT=0.0
    STOPPED_SAMPLE_WEIGHT=1.7
    HAZARD_SAMPLE_WEIGHT=3.2
    LAUNCH_SAMPLE_WEIGHT=2.3
    RELEASE_SAMPLE_WEIGHT=2.1
    STOP_LOSS_WEIGHT=0.12
    STOP_SPEED_CEILING_LOSS_WEIGHT=0.60
    SPEED_FLOOR_LOSS_WEIGHT=0.055
    LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=0.09
    RELEASE_SPEED_FLOOR_LOSS_WEIGHT=0.09
    STOP_STATE_AUX_LOSS_WEIGHT=0.12
    STOP_REASON_AUX_LOSS_WEIGHT=0.06
    TRAJ_SMOOTH_LOSS_WEIGHT=0.030
    SPEED_SMOOTH_LOSS_WEIGHT=0.016
    "$@"
    bash configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh
  )
  local command_string=""
  printf -v command_string "%q " "${cmd[@]}"
  tmux new-session -d -s "$session" -c "$PWD" -- bash -lc "$command_string > $(printf '%q' "$log") 2>&1"
}

# A. Residual-only: the cleanest test of bounded output correction.
launch residual_frozen_b24 0 \
  SEED=71 \
  LR=1.5e-5 \
  LORA_RANK=0 \
  CONTROL_LOSS_WEIGHT=0.75 \
  OUTPUT_RESIDUAL_HIDDEN_DIM=256 \
  OUTPUT_RESIDUAL_CHECKPOINT_SCALE=0.60 \
  OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE=1.25 \
  OUTPUT_RESIDUAL_GATE_BIAS=-2.3 \
  SELECTION_METRIC=controller_closed_loop_proxy \
  SELECTION_MODE=min

# B. Tiny LoRA plus residual: closest successor to action_cl1.
launch residual_lora4_b24 1 \
  SEED=72 \
  LR=1.2e-5 \
  LORA_RANK=4 \
  LORA_ALPHA=8 \
  CONTROL_LOSS_WEIGHT=0.85 \
  OUTPUT_RESIDUAL_HIDDEN_DIM=256 \
  OUTPUT_RESIDUAL_CHECKPOINT_SCALE=0.70 \
  OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE=1.50 \
  OUTPUT_RESIDUAL_GATE_BIAS=-2.0 \
  SELECTION_METRIC=controller_closed_loop_proxy \
  SELECTION_MODE=min

# C. Stronger action correction: more capacity, still bounded.
launch residual_lora8_go_b24 2 \
  SEED=73 \
  LR=1.0e-5 \
  LORA_RANK=8 \
  LORA_ALPHA=16 \
  CONTROL_LOSS_WEIGHT=1.05 \
  GO_PROGRESS_RATIO=0.70 \
  CONTROLLER_THROTTLE_CLOSE_THRESHOLD=0.16 \
  OUTPUT_RESIDUAL_HIDDEN_DIM=384 \
  OUTPUT_RESIDUAL_CHECKPOINT_SCALE=0.85 \
  OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE=1.75 \
  OUTPUT_RESIDUAL_GATE_BIAS=-1.8 \
  SELECTION_METRIC=controller_closed_loop_proxy \
  SELECTION_MODE=min

# D. Conservative safety variant: smaller trajectory delta, stronger stop/hold.
launch residual_lora4_safe_b24 3 \
  SEED=74 \
  LR=8e-6 \
  LORA_RANK=4 \
  LORA_ALPHA=8 \
  CONTROL_LOSS_WEIGHT=0.90 \
  HAZARD_SAMPLE_WEIGHT=4.0 \
  STOPPED_SAMPLE_WEIGHT=2.0 \
  STOP_SPEED_CEILING_LOSS_WEIGHT=0.80 \
  OUTPUT_RESIDUAL_HIDDEN_DIM=256 \
  OUTPUT_RESIDUAL_CHECKPOINT_SCALE=0.45 \
  OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE=1.20 \
  OUTPUT_RESIDUAL_GATE_BIAS=-2.4 \
  SELECTION_METRIC=controller_closed_loop_proxy \
  SELECTION_MODE=min

echo "sessions:"
tmux ls | grep -E 't2d_residual_(frozen|lora4|lora8)' || true
