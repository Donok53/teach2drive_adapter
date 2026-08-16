#!/usr/bin/env bash
# v5 = v4 paired recipe with STRONGER canonical-feature alignment (single-direction change).
#
# Confirmed diagnosis (test1: TF++ zero-shot + canonical rig + Tesla recovers what v4/shifted fails):
#   The root cause of v4's shortfall is the SHIFTED camera (+2.75m fwd): (a) peripheral info loss
#   + (b) camera<->lidar geometry mismatch (2.55m) breaking TF++'s learned fusion correspondence.
#   Behaviour fixes (v4c residual / v4d perturbation / v4e speed-residual) ALL made it worse -> the
#   problem is PERCEPTION, not action. v4's feature-distill (pull shifted features -> canonical
#   teacher features) is the ONE correct mechanism, but v4 ran it too weakly (drift=0.12, rank=8)
#   to invert a 2.75m reprojection.
# v5 strengthens exactly that, nothing else:
#   --feature-drift-loss-weight 0.12 -> 0.5   (make canonical-feature matching a primary objective)
#   --lora-rank 8 -> 16, --lora-alpha 16 -> 32 (capacity for the shifted->canonical transform)
# NO perturbation, NO output-residual. Everything else identical to v4 (same seed 91).
# NOTE: evaluate v5 with MISSION_TIMEOUT_SEC=500 (fair vs native), and pick the epoch by
# closed-loop, not val (val != driving throughout this project).
set -u

RUN_DIR=${RUN_DIR:-/data/users/byeongjae/runs/train_v5_paired_strong_align}
OUT_DIR="${RUN_DIR}/train"
mkdir -p "$OUT_DIR"

DATA_ROOT=/data/users/byeongjae
DATASETS_ROOT=/data/users/byeongjae/datasets
CODE_ROOT=/data/users/byeongjae/code

cd "${CODE_ROOT}/teach2drive_adapter"

echo "=== v5 = v4 + STRONG canonical alignment (drift 0.5, lora 16/32) start $(date '+%F %T') ==="
echo "run_dir=${RUN_DIR}"
echo "commit=$(git rev-parse --short HEAD 2>/dev/null)"

PYTHONPATH="${CODE_ROOT}/teach2drive_adapter:${CODE_ROOT}/carla_garage/team_code:${PYTHONPATH:-}" \
python -m teach2drive_adapter.train_transfuserpp_task_feature_adapter \
  --index "${DATASETS_ROOT}/t2d_paired_shifted_3h_index.npz" \
  --episode-root-override "${DATASETS_ROOT}/t2d_paired_shifted_3h" \
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
  --extrinsic-aware \
  --source-profile front_triplet_shifted \
  --extrinsic-hidden-dim 64 \
  --extrinsic-dropout 0.0 \
  --hidden-channels 0 \
  --blocks 2 \
  --dropout 0.0 \
  --stage-adapter-layers all \
  --stage-adapter-modalities all \
  --stage-feature-adapter-blend 1.0 \
  --fusion-adapter-blend 1.0 \
  --lora-rank 16 \
  --lora-alpha 32.0 \
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
  --speed-loss-weight 0.8 \
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
  --feature-drift-loss-weight 0.5 \
  --output-prior-xy-loss-weight 0.0 \
  --output-prior-speed-loss-weight 0.0 \
  --aux-hidden-dim 256 \
  --control-loss-weight 0.9 \
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
  --teacher-view-root "${DATASETS_ROOT}/pdm_lite_tesla_paired_3h/data" \
  --teacher-view-dirname rgb_canonical \
  --step-log-every 50

echo "=== v5 training finished $(date '+%F %T') rc=$? ==="
