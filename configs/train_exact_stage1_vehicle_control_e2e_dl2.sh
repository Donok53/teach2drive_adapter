#!/usr/bin/env bash
set -euo pipefail

RUN_NAME=${RUN_NAME:-train_exact_stage1_vehicle_control_e2e_gpu0}
RUN_ROOT=${RUN_ROOT:-/data/dataset/byeongjae/runs}
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
CACHE=${CACHE:-${RUN_DIR}/canonical_tfpp_prior_with_control.npz}
INDEX=${INDEX:-/data/dataset/byeongjae/datasets/t2d_paired_shifted_3h_tfpp_exact_index.npz}
EPISODE_ROOT=${EPISODE_ROOT:-/data/dataset/byeongjae/datasets/t2d_paired_shifted_3h_tfpp_exact}
TEACHER_VIEW_ROOT=${TEACHER_VIEW_ROOT:-/data/dataset/byeongjae/datasets/pdm_lite_tesla_paired_3h/data}
TEACHER_VIEW_DIRNAME=${TEACHER_VIEW_DIRNAME:-rgb_canonical}
GARAGE_ROOT=${GARAGE_ROOT:-/data/users/byeongjae/code/carla_garage}
TEAM_CONFIG=${TEAM_CONFIG:-/data/dataset/byeongjae/checkpoints/transfuserpp/pretrained_models/all_towns}

mkdir -p "${RUN_DIR}/train"

if [[ ! -s "${CACHE}" ]]; then
  echo "=== cache canonical TF++ prior + Tesla expert controls $(date '+%F %T') ==="
  python -m teach2drive_adapter.cache_transfuserpp_prior \
    --index "${INDEX}" \
    --episode-root-override "${EPISODE_ROOT}" \
    --output "${CACHE}" \
    --garage-root "${GARAGE_ROOT}" \
    --team-config "${TEAM_CONFIG}" \
    --cameras front \
    --tfpp-camera front \
    --teacher-view-root "${TEACHER_VIEW_ROOT}" \
    --teacher-view-dirname "${TEACHER_VIEW_DIRNAME}" \
    --command-mode target_angle \
    --image-size 1024 512 \
    --lidar-size 256 \
    --ensemble-all-checkpoints \
    --batch-size "${CACHE_BATCH_SIZE:-24}" \
    --num-workers "${NUM_WORKERS:-4}" \
    --log-every 20
fi

echo "=== train post-PID turn-gated Tesla steer adapter $(date '+%F %T') ==="
python -m teach2drive_adapter.train_tfpp_vehicle_steer_adapter \
  --cache "${CACHE}" \
  --out-dir "${RUN_DIR}/train" \
  --epochs "${EPOCHS:-30}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE:-6}" \
  --batch-size "${BATCH_SIZE:-1024}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-3.0e-4}" \
  --hidden-dim "${HIDDEN_DIM:-128}" \
  --max-delta "${MAX_DELTA:-0.20}" \
  --turn-sample-weight "${TURN_SAMPLE_WEIGHT:-6.0}" \
  --straight-identity-weight "${STRAIGHT_IDENTITY_WEIGHT:-4.0}" \
  --residual-magnitude-weight "${RESIDUAL_MAGNITUDE_WEIGHT:-0.25}" \
  --seed "${SEED:-91}"
