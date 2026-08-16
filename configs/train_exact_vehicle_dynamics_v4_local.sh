#!/usr/bin/env bash
set -euo pipefail

ADAPTER_ROOT=${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}
RUN_DIR=${RUN_DIR:-${ADAPTER_ROOT}/runs/train_exact_vehicle_dynamics_v4_local}
CACHE=${CACHE:-${ADAPTER_ROOT}/runs/cache_stage1_vehicle_control/canonical_tfpp_prior_with_control.npz}
INDEX=${INDEX:-/home/byeongjae/dataset/byeongjae/datasets/t2d_paired_shifted_3h_tfpp_exact_index.npz}
EPISODE_ROOT=${EPISODE_ROOT:-/home/byeongjae/dataset/byeongjae/datasets/t2d_pdm_lite_front_triplet_shifted_3h}

mkdir -p "${RUN_DIR}/train"
python -m teach2drive_adapter.train_tfpp_yaw_horizon_calibrator_v4 \
  --cache "${CACHE}" \
  --index "${INDEX}" \
  --episode-root "${EPISODE_ROOT}" \
  --out-dir "${RUN_DIR}/train" \
  --minimum-gain "${MINIMUM_GAIN:-0.85}" \
  --maximum-gain "${MAXIMUM_GAIN:-1.10}" \
  --identity-weight "${IDENTITY_WEIGHT:-2.0}" \
  --turn-weight "${TURN_WEIGHT:-2.0}" \
  --epochs "${EPOCHS:-300}" \
  --lr "${LR:-0.03}" \
  --seed "${SEED:-91}"
