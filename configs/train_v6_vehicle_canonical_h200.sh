#!/usr/bin/env bash
# v6 = VEHICLE adaptation on the CANONICAL camera (isolate the vehicle axis).
#   input  = canonical camera (Tesla, rgb_canonical)  -> no camera-domain gap
#   target = Tesla expert trajectory/speed            -> learn Tesla behavior directly
#   NO feature-distill (camera already canonical), NO extrinsic-aware (native input).
# Baseline to beat = test1 (Tesla+canonical, no adapter) = 78.1 composed; ceiling = native 94.4.
# Params swept via env: RUN_TAG, LORA_RANK, LORA_ALPHA, SPEED_W, CONTROL_W, USE_RESIDUAL.
set -u
RUN_TAG=${RUN_TAG:-va}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
SPEED_W=${SPEED_W:-0.8}
CONTROL_W=${CONTROL_W:-0.9}
USE_RESIDUAL=${USE_RESIDUAL:-0}

RUN_DIR=${RUN_DIR:-/data/users/byeongjae/runs/train_v6_vehicle_canonical_${RUN_TAG}}
OUT_DIR="${RUN_DIR}/train"
mkdir -p "$OUT_DIR"
DATA_ROOT=/data/users/byeongjae
DATASETS_ROOT=/data/users/byeongjae/datasets
CODE_ROOT=/data/users/byeongjae/code
cd "${CODE_ROOT}/teach2drive_adapter"

RESIDUAL_ARGS=""
if [ "$USE_RESIDUAL" = "1" ]; then
  RESIDUAL_ARGS="--output-residual --output-residual-checkpoint-scale 0.0 --output-residual-speed-logit-scale 2.0 --output-residual-gate-bias -1.0 --output-residual-hidden-dim 256"
fi

echo "=== v6 vehicle-adapt(canonical) tag=$RUN_TAG rank=$LORA_RANK alpha=$LORA_ALPHA speed=$SPEED_W ctrl=$CONTROL_W residual=$USE_RESIDUAL start $(date '+%F %T') ==="
echo "run_dir=${RUN_DIR}"

PYTHONPATH="${CODE_ROOT}/teach2drive_adapter:${CODE_ROOT}/carla_garage/team_code:${PYTHONPATH:-}" \
python -m teach2drive_adapter.train_transfuserpp_task_feature_adapter \
  --index "${DATASETS_ROOT}/t2d_paired_shifted_3h_index.npz" \
  --episode-root-override "${DATASETS_ROOT}/t2d_paired_canonical_3h" \
  --out-dir "${OUT_DIR}" \
  --garage-root "${CODE_ROOT}/carla_garage" \
  --team-config "${DATA_ROOT}/checkpoints/transfuserpp/pretrained_models/all_towns" \
  --checkpoint '' \
  --init-checkpoint '' \
  --cameras front \
  --tfpp-camera front \
  --command-mode target_angle \
  --image-size 640 360 \
  --lidar-size 128 \
  --hidden-channels 0 \
  --blocks 2 \
  --dropout 0.0 \
  --stage-adapter-layers all \
  --stage-adapter-modalities all \
  --stage-feature-adapter-blend 1.0 \
  --fusion-adapter-blend 1.0 \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout 0.02 \
  --lora-include '^join\.,^checkpoint_decoder\.(encoder|decoder)\.,^target_speed_network\.' \
  --lora-exclude '' \
  --epochs 20 \
  --early-stop-patience 8 \
  --early-stop-min-delta 0.0 \
  --selection-metric loss \
  --selection-mode min \
  --save-epoch-checkpoints \
  --epoch-checkpoint-dir epoch_checkpoints \
  --batch-size 24 \
  --num-workers 4 \
  --lr 1e-05 \
  --weight-decay 0.0001 \
  --val-ratio 0.15 \
  --seed 91 \
  --speed-dim 4 \
  --xy-loss-weight 0.55 \
  --yaw-loss-weight 0.03 \
  --speed-loss-weight "$SPEED_W" \
  --traj-smooth-loss-weight 0.03 \
  --speed-smooth-loss-weight 0.02 \
  --speed-floor-loss-weight 0.18 \
  --speed-floor-mps 0.8 \
  --speed-floor-target-threshold 2.0 \
  --stop-speed-ceiling-loss-weight 0.2 \
  --stop-speed-ceiling-mps 0.5 \
  --stop-speed-target-threshold 0.5 \
  --stop-progress-ceiling-m 1.0 \
  --go-progress-ratio 0.5 \
  --stop-loss-weight 0.05 \
  --feature-drift-loss-weight 0.0 \
  --output-prior-xy-loss-weight 0.0 \
  --output-prior-speed-loss-weight 0.0 \
  --aux-hidden-dim 256 \
  --control-loss-weight "$CONTROL_W" \
  --stop-state-aux-loss-weight 0.15 \
  --stop-reason-aux-loss-weight 0.1 \
  --moving-speed-threshold 1.0 \
  --moving-sample-weight 1.4 \
  --stopped-sample-weight 0.7 \
  --hazard-stop-reasons traffic_light,stop_sign,front_vehicle,junction_yield \
  --hazard-sample-weight 1.5 \
  --launch-current-speed-threshold 0.8 \
  --launch-target-speed-threshold 2.0 \
  --launch-sample-weight 4.0 \
  --launch-speed-floor-loss-weight 0.3 \
  --launch-speed-floor-mps 1.2 \
  --release-target-speed-threshold 1.0 \
  --release-sample-weight 3.5 \
  --release-speed-floor-loss-weight 0.3 \
  --release-speed-floor-mps 1.2 \
  --controller-steer-close-threshold 0.15 \
  --controller-throttle-close-threshold 0.2 \
  --controller-go-throttle-threshold 0.05 \
  --grad-clip 1.0 \
  --max-train-samples 0 \
  --max-val-samples 0 \
  $RESIDUAL_ARGS \
  --step-log-every 50

echo "=== v6 $RUN_TAG finished $(date '+%F %T') rc=$? ==="
