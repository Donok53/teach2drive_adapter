#!/usr/bin/env bash
set -euo pipefail

# Canonical-sensor Tesla vehicle adaptation. TF++ perception/planning and the
# stock longitudinal controller remain frozen; only a bounded, turn-gated
# post-PID lateral residual is trained from Tesla expert controls.

RUN_NAME=${RUN_NAME:-train_exact_stage1_vehicle_inverse_dynamics_v2_gpu0}
RUN_ROOT=${RUN_ROOT:-/data/dataset/byeongjae/runs}
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
SOURCE_CACHE=${SOURCE_CACHE:-${RUN_ROOT}/train_exact_stage1_vehicle_control_e2e_v1_gpu0/canonical_tfpp_prior_with_control.npz}
EPISODE_ROOT=${EPISODE_ROOT:-/data/dataset/byeongjae/datasets/t2d_paired_shifted_3h_tfpp_exact}

mkdir -p "${RUN_DIR}/train"

if [[ ! -s "${SOURCE_CACHE}" ]]; then
  echo "Missing canonical TF++ prior cache: ${SOURCE_CACHE}" >&2
  exit 2
fi

echo "=== train dynamics-conditioned bounded Tesla steer adapter $(date '+%F %T') ==="
echo "run_dir=${RUN_DIR} cache=${SOURCE_CACHE} episode_root=${EPISODE_ROOT}"

python -m teach2drive_adapter.train_tfpp_vehicle_steer_adapter \
  --cache "${SOURCE_CACHE}" \
  --episode-root "${EPISODE_ROOT}" \
  --out-dir "${RUN_DIR}/train" \
  --epochs "${EPOCHS:-24}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE:-4}" \
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA:-2.0e-4}" \
  --batch-size "${BATCH_SIZE:-1024}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-2.0e-4}" \
  --hidden-dim "${HIDDEN_DIM:-128}" \
  --dropout "${DROPOUT:-0.05}" \
  --adapter-mode gain \
  --minimum-gain "${MINIMUM_GAIN:-0.10}" \
  --maximum-gain "${MAXIMUM_GAIN:-1.20}" \
  --minimum-base-steer "${MINIMUM_BASE_STEER:-0.05}" \
  --minimum-expert-steer "${MINIMUM_EXPERT_STEER:-0.05}" \
  --max-delta "${MAX_DELTA:-0.12}" \
  --turn-threshold "${TURN_THRESHOLD:-0.08}" \
  --full-turn-threshold "${FULL_TURN_THRESHOLD:-0.16}" \
  --minimum-turn-loss-gate "${MINIMUM_TURN_LOSS_GATE:-0.05}" \
  --straight-gate-threshold "${STRAIGHT_GATE_THRESHOLD:-0.05}" \
  --opposite-steer-deadband "${OPPOSITE_STEER_DEADBAND:-0.05}" \
  --straight-identity-weight "${STRAIGHT_IDENTITY_WEIGHT:-8.0}" \
  --residual-magnitude-weight "${RESIDUAL_MAGNITUDE_WEIGHT:-0.50}" \
  --selection-straight-weight "${SELECTION_STRAIGHT_WEIGHT:-1.0}" \
  --minimum-speed "${MINIMUM_SPEED:-0.5}" \
  --use-yaw-rate \
  --seed "${SEED:-91}"
