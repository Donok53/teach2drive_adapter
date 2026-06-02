#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

# Reuse the current four action-closed-loop adapter recipes, but swap the
# training data to the completed benchmark-aligned 8-town Tesla target dataset.

PY="${PY:-/home/jovyan_venv/.venv/torch2.1.2-py3.10-cuda11.8/bin/python}"
BOOTSTRAP_ROOT="${BOOTSTRAP_ROOT:-$HOME/teach2drive/workspace/teach2drive_bootstrap}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-$HOME/dataset/byeongjae/datasets/t2d_tesla_benchmix8_front_triplet_target_3h}"
PREP_ROOT="${PREP_ROOT:-$HOME/dataset/byeongjae/runs/tfpp_tesla_benchmix8_front_triplet_target_3h_prepared}"
RUN_ROOT="${RUN_ROOT:-$HOME/dataset/byeongjae/runs/benchmix8_target3h_action_cl}"
LOG_DIR="${LOG_DIR:-$HOME/teach2drive/logs/benchmix8_target3h_action_cl}"
PROFILE="${PROFILE:-front_triplet_shifted}"
CAMERAS="${CAMERAS:-left,front,right}"
TARGET_VIEW="${TARGET_VIEW:-$PREP_ROOT/profile_views/$PROFILE}"
TARGET_INDEX="${TARGET_INDEX:-$PREP_ROOT/indexes/${PROFILE}_index.npz}"
EXPORT_OVERWRITE="${EXPORT_OVERWRITE:-1}"
INDEX_OVERWRITE="${INDEX_OVERWRITE:-1}"
SKIP_PREP="${SKIP_PREP:-0}"

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
        --seed 61
    )
  fi
fi

echo "=== launch four benchmix8 action closed-loop trainings"
echo "TARGET_VIEW=$TARGET_VIEW"
echo "TARGET_INDEX=$TARGET_INDEX"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_DIR=$LOG_DIR"

env \
  PY="$PY" \
  RUN_ROOT="$RUN_ROOT" \
  VIEW_ROOT="$PREP_ROOT/profile_views" \
  INDEX_DIR="$PREP_ROOT/indexes" \
  TARGET_VIEW="$TARGET_VIEW" \
  TARGET_INDEX="$TARGET_INDEX" \
  LOG_DIR="$LOG_DIR" \
  bash configs/train_tfpp_town13_action_closedloop4_server.sh
