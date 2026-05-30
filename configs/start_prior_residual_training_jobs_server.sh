#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="${LOG_DIR:-$HOME/teach2drive/logs}"
TORCH_ENV="${TORCH_ENV:-$HOME/.venv/torch2.1.2-py3.10-cuda11.8}"
PYTHON_BIN="${PYTHON_BIN:-$TORCH_ENV/bin/python}"
mkdir -p "$LOG_DIR"

start_job() {
  local session="$1"
  local gpu="$2"
  local variant="$3"
  local run_name="$4"
  local log_file="$5"

  tmux kill-session -t "$session" 2>/dev/null || true
  rm -f "$log_file"
  tmux new-session -d -s "$session" bash -lc \
    "cd '$PWD' && PATH='$TORCH_ENV/bin':\$PATH PY='$PYTHON_BIN' CUDA_VISIBLE_DEVICES='$gpu' VARIANT='$variant' RUN_NAME='$run_name' bash configs/train_tfpp_tesla_town13_prior_residual_variant_server.sh > '$log_file' 2>&1"
  echo "started $session gpu=$gpu variant=$variant run=$run_name log=$log_file"
}

start_job train_prior_feature_g0 0 feature prior_feature_g0_restart "$LOG_DIR/train_prior_feature_g0_restart.log"
start_job train_prior_lora2_g1 1 lora2 prior_lora2_g1_restart "$LOG_DIR/train_prior_lora2_g1_restart.log"
start_job train_prior_lora4_g2 2 lora4 prior_lora4_g2_restart "$LOG_DIR/train_prior_lora4_g2_restart.log"
start_job train_prior_strict_g3 3 strict prior_strict_g3_restart "$LOG_DIR/train_prior_strict_g3_restart.log"

tmux ls
