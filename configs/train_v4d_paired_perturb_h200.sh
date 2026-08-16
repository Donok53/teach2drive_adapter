#!/usr/bin/env bash
# v4d = v4 paired recipe + recovery-perturbation augmentation ON (single-variable change).
#
# v4 (train_v4_paired_teacher_h200.sh) = feature-distill(0.12) + control(0.9) imitation on
# the canonical teacher-view, LoRA rank8, backbone FROZEN, output-residual OFF -> 70.5 composed
# (10/20 pass). Closed-loop diff vs native TF++ (14/20): our 7 losses are ALL junction turns,
# failing mid-turn (collision/off-road, "hesitant" limp). Root cause = the 3h paired data is
# EXPERT-TRAJECTORY-ONLY, so the adapter never learned to recover from an off-heading state;
# TF++ itself is robust there because it was trained with augmentation. This run RESTORES that
# capability via the built-in ChauffeurNet-style yaw recovery perturbation:
#   - camera panned by random psi (+-6 deg, crop-shift ~ yaw), focal_px=224 matches the
#     640-wide / 110-deg-FOV input (=320/tan(55deg)); expert trajectory + nav target_point are
#     re-expressed in the perturbed ego frame -> model learns to steer back onto the path.
# Everything else IDENTICAL to v4 (same seed 91) so the perturbation is the ONLY variable.
# output-residual stays OFF (v4c proved its lateral component corrupts clean straights).
# xmodal alignment intentionally left OFF here -> follow-up v4e if turns still need more.
set -u

RUN_DIR=${RUN_DIR:-/data/dataset/byeongjae/runs/train_v4d_paired_perturb}
OUT_DIR="${RUN_DIR}/train"
mkdir -p "$OUT_DIR"

DATA_ROOT=/data/dataset/byeongjae
DATASETS_ROOT=/data/users/byeongjae/datasets   # datasets moved here 2026-07-27 (same fs)
CODE_ROOT=/data/users/byeongjae/code

cd "${CODE_ROOT}/teach2drive_adapter"

echo "=== v4d = v4 + recovery-perturbation (prob=0.4, psi=6deg, focal=224) start $(date '+%F %T') ==="
echo "run_dir=${RUN_DIR}"
echo "commit=$(git rev-parse --short HEAD 2>/dev/null)"

PYTHONPATH="${CODE_ROOT}/teach2drive_adapter:${CODE_ROOT}/carla_garage/team_code:${PYTHONPATH:-}" \
T2D_PERTURB_PROB=0.4 \
T2D_PERTURB_PSI_MAX_DEG=6 \
T2D_PERTURB_FOCAL_PX=224 \
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
  --teacher-view-root "${DATASETS_ROOT}/pdm_lite_tesla_paired_3h/data" \
  --teacher-view-dirname rgb_canonical \
  --step-log-every 50

echo "=== v4d training finished $(date '+%F %T') rc=$? ==="
