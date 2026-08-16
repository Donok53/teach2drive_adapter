#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-train_exact_vehicle_dynamics_v3_gpu0}
RUN_ROOT=${RUN_ROOT:-/data/dataset/byeongjae/runs}
EPISODE_ROOT=${EPISODE_ROOT:-/data/dataset/byeongjae/datasets/t2d_paired_shifted_3h_tfpp_exact}
V1_CHECKPOINT=${V1_CHECKPOINT:-${RUN_ROOT}/train_exact_vehicle_dynamics_tesla_v1_gpu0/train/best_model.pt}
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"

mkdir -p "${RUN_DIR}/train"
echo "=== fit lag-aware multi-horizon Tesla yaw dynamics v3 $(date '+%F %T') ==="
echo "run_dir=${RUN_DIR} episode_root=${EPISODE_ROOT}"

python -m teach2drive_adapter.train_tfpp_vehicle_dynamics_v3 \
  --episode-root "${EPISODE_ROOT}" \
  --out-dir "${RUN_DIR}/train" \
  --v1-checkpoint "${V1_CHECKPOINT}" \
  --smooth-frames "${SMOOTH_FRAMES:-3}" \
  --minimum-speed "${MINIMUM_SPEED:-1.0}" \
  --turn-weight "${TURN_WEIGHT:-4.0}" \
  --turn-scale "${TURN_SCALE:-0.20}" \
  --ridge "${RIDGE:-1.0e-3}" \
  --irls-iterations "${IRLS_ITERATIONS:-5}" \
  --rollout-steps "${ROLLOUT_STEPS:-3}" \
  --seed "${SEED:-91}"
