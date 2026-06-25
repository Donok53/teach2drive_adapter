#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SESSION="${SESSION:-t2d_safe_ctrl_conservative_queue}"
export SAFE_CTRL_RECIPE_SET="${SAFE_CTRL_RECIPE_SET:-conservative}"
export SAFE_CTRL_GPUS="${SAFE_CTRL_GPUS:-0,2}"
export SAFE_CTRL_EPOCH1_THRESHOLD="${SAFE_CTRL_EPOCH1_THRESHOLD:-6.8}"
export SAFE_CTRL_EPOCH2_THRESHOLD="${SAFE_CTRL_EPOCH2_THRESHOLD:-6.2}"
export PROMOTE_EPOCHS="${PROMOTE_EPOCHS:-10}"
export RUN_ROOT="${RUN_ROOT:-$HOME/dataset/byeongjae/runs/benchmix8_safe_ctrl_conservative_queue_target_only}"
LOG_DIR="${LOG_DIR:-$HOME/teach2drive/logs/benchmix8_safe_ctrl_conservative_queue_target_only}"
export LOG_DIR
LOG="${LOG:-$LOG_DIR/scheduler.log}"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session already exists: $SESSION"
  echo "log: $LOG"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$PWD" -- bash -lc \
  "python3 configs/run_safe_ctrl_proxy_queue.py > $(printf '%q' "$LOG") 2>&1"

echo "started scheduler: $SESSION"
echo "log: $LOG"
echo "decisions: $LOG_DIR/safe_ctrl_queue_decisions.jsonl"
