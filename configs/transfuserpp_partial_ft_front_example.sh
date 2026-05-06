#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-python}
INDEX=${INDEX:-"$HOME/teach2drive/datasets/token_pseudo_rule_multicam_index.npz"}
OUT=${OUT:-"runs/tfpp_partial_ft_front"}
GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}
CHECKPOINT=${CHECKPOINT:-""}
EPISODE_ROOT=${EPISODE_ROOT:-"$HOME/teach2drive/datasets/teach2drive/town10_3cam_640x360"}
TRAIN_PRIOR=${TRAIN_PRIOR:-"join,transformer,decoder"}

mkdir -p "$OUT"

ARGS=(
  -m teach2drive_adapter.train_transfuserpp_adapter
  --index "$INDEX"
  --out-dir "$OUT"
  --garage-root "$GARAGE_ROOT"
  --team-config "$TEAM_CONFIG"
  --episode-root-override "$EPISODE_ROOT"
  --cameras front
  --tfpp-camera front
  --command-mode target_angle
  --epochs 8
  --batch-size 16
  --num-workers 8
  --lr 1e-4
  --prior-lr 1e-5
  --train-prior-substrings "$TRAIN_PRIOR"
  --data-parallel
)

if [[ -n "$CHECKPOINT" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

PYTHONUNBUFFERED=1 "$PY" "${ARGS[@]}" 2>&1 | tee "$OUT/train.log"
