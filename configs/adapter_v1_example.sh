#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-python}
INDEX=${INDEX:-"$HOME/code/teach2drive_bootstrap/runs/town10_3cam_640x360_tokens/token_pseudo_rule_multicam_index.npz"}
OUT=${OUT:-"runs/adapter_v1"}
PRETRAINED=${PRETRAINED:-""}

mkdir -p "$OUT"

ARGS=(
  -m teach2drive_adapter.train_adapter
  --index "$INDEX"
  --out-dir "$OUT"
  --mode adapter
  --epochs 20
  --batch-size 32
  --lr 3e-4
  --embed-dim 256
  --adapter-dim 64
  --image-size 320 180
  --num-workers 8
  --step-log-every 200
)

if [[ -n "$PRETRAINED" ]]; then
  ARGS+=(--pretrained "$PRETRAINED")
fi

PYTHONUNBUFFERED=1 "$PY" "${ARGS[@]}" 2>&1 | tee "$OUT/train.log"

