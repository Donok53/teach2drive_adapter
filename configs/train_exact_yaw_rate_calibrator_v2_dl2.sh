#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-train_exact_yaw_rate_calibrator_v2_gpu0}
RUN_ROOT=${RUN_ROOT:-/data/dataset/byeongjae/runs}
SOURCE_CACHE=${SOURCE_CACHE:-${RUN_ROOT}/train_exact_stage1_vehicle_control_e2e_v1_gpu0/canonical_tfpp_prior_with_control.npz}
INDEX=${INDEX:-/data/dataset/byeongjae/datasets/t2d_paired_shifted_3h_tfpp_exact_index.npz}
EPISODE_ROOT=${EPISODE_ROOT:-/data/dataset/byeongjae/datasets/t2d_paired_shifted_3h_tfpp_exact}
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"

mkdir -p "${RUN_DIR}/train"
echo "=== fit bounded geometric yaw-rate calibrator v2 $(date '+%F %T') ==="
python -m teach2drive_adapter.train_tfpp_yaw_rate_calibrator \
  --cache "${SOURCE_CACHE}" \
  --index "${INDEX}" \
  --episode-root "${EPISODE_ROOT}" \
  --out-dir "${RUN_DIR}/train" \
  --minimum-gain "${MINIMUM_GAIN:-0.75}" \
  --maximum-gain "${MAXIMUM_GAIN:-1.15}" \
  --identity-weight "${IDENTITY_WEIGHT:-1.0}" \
  --turn-weight "${TURN_WEIGHT:-2.0}" \
  --epochs "${EPOCHS:-300}" \
  --lr "${LR:-0.03}" \
  --seed "${SEED:-91}"
