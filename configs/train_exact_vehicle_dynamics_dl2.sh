#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-train_exact_vehicle_dynamics_tesla_v1_gpu0}
RUN_ROOT=${RUN_ROOT:-/data/dataset/byeongjae/runs}
EPISODE_ROOT=${EPISODE_ROOT:-/data/dataset/byeongjae/datasets/t2d_paired_shifted_3h_tfpp_exact}
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"

mkdir -p "${RUN_DIR}/train"
echo "=== fit Tesla steer-to-yaw-rate dynamics $(date '+%F %T') ==="
echo "run_dir=${RUN_DIR} episode_root=${EPISODE_ROOT}"

python -m teach2drive_adapter.train_tfpp_vehicle_dynamics \
  --episode-root "${EPISODE_ROOT}" \
  --out-dir "${RUN_DIR}/train" \
  --smooth-frames "${SMOOTH_FRAMES:-3}" \
  --minimum-speed "${MINIMUM_SPEED:-1.0}" \
  --turn-weight "${TURN_WEIGHT:-4.0}" \
  --turn-scale "${TURN_SCALE:-0.20}" \
  --ridge "${RIDGE:-1.0e-3}" \
  --irls-iterations "${IRLS_ITERATIONS:-5}" \
  --seed "${SEED:-91}"
