#!/usr/bin/env bash
# Re-run of the v3 recipe on the GIST 3h dataset with the ORIGINAL (75-score) settings.
#
# Why this exists: train_v3_gist3h scored 14.09 composed (20 missions / 180s) while the
# reference v3 model scored ~50-75. Diffing the two checkpoints' metadata showed the new
# run had drifted from the recipe that produced the good model in three ways:
#
#   1. --unfreeze-include '^backbone\.'  was ADDED  -> 121M backbone params were trained on
#      only 3h (33k samples) of data. Suspected catastrophic forgetting of the pretrained
#      TF++ representations: val loss improved (1.628 vs 1.669) while closed-loop driving
#      collapsed. THIS IS THE PRIMARY SUSPECT -> removed here (backbone stays frozen).
#   2. --lora-include dropped '^checkpoint_decoder\.(encoder|decoder)\.' -> restored.
#   3. --batch-size 12 instead of 24 -> restored.
#
# Everything else (loss weights, lr, seed, val split, adapter config) already matched.
# Runs on H200 GPU7 inside the t2d-train:cu118 container.
set -u

RUN_DIR=${RUN_DIR:-/data/dataset/byeongjae/runs/train_v3_gist3h_earlyadapter}
OUT_DIR="${RUN_DIR}/train"
mkdir -p "$OUT_DIR"

DATA_ROOT=/data/dataset/byeongjae
CODE_ROOT=/data/users/byeongjae/code

cd "${CODE_ROOT}/teach2drive_adapter"

echo "=== v3 recipe on GIST 3h, backbone FROZEN + stage-adapter early:2 (front fusion layers only) start $(date '+%F %T') ==="
echo "run_dir=${RUN_DIR}"
echo "commit=$(git rev-parse --short HEAD 2>/dev/null)"

PYTHONPATH="${CODE_ROOT}/teach2drive_adapter:${CODE_ROOT}/carla_garage/team_code:${PYTHONPATH:-}" \
python -m teach2drive_adapter.train_transfuserpp_task_feature_adapter \
  --index "${DATA_ROOT}/datasets/t2d_gist3h_index.npz" \
  --episode-root-override "${DATA_ROOT}/datasets/t2d_pdm_lite_front_triplet_shifted_3h" \
  --out-dir "${OUT_DIR}" \
  --garage-root "${CODE_ROOT}/carla_garage" \
  --team-config "${DATA_ROOT}/checkpoints/transfuserpp/pretrained_models/all_towns" \
  --checkpoint '' \
  --init-checkpoint '' \
  --cameras left,front,right \
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
  --stage-adapter-layers early:2 \
  --stage-adapter-modalities all \
  --stage-feature-adapter-blend 1.0 \
  --fusion-adapter-blend 1.0 \
  --lora-rank 8 \
  --lora-alpha 16.0 \
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
  --feature-drift-loss-weight 0.12 \
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
  --step-log-every 50

echo "=== training finished $(date '+%F %T') rc=$? ==="
