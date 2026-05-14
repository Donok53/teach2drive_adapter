#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# v3.5 preset: teacher-following adaptation with hazard-conditional speed.
# Normal driving imitates canonical TransFuser++, while traffic-light/stop/yield
# frames use the collected expert speed targets so the policy can learn when not
# to follow the fast teacher.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}

export PREV_WORK_ROOT=${PREV_WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_shifted_adapter_motion_safe"}
export V3_WORK_ROOT=${V3_WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_shifted_adapter_motion_safe_e2e_v3_teacher_distill"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_shifted_adapter_motion_safe_e2e_v35_hazard_teacher"}
export VIEW_ROOT=${VIEW_ROOT:-"$WORK_ROOT/profile_views"}
export INDEX_DIR=${INDEX_DIR:-"$WORK_ROOT/indexes"}

export CACHE=${CACHE:-"$PREV_WORK_ROOT/cache/tfpp_ego_prior_cache.npz"}
export SHIFT_CACHE=${SHIFT_CACHE:-"$V3_WORK_ROOT/cache/front_triplet_shifted_prior_cache.npz"}
export TRAIN_CACHE=${TRAIN_CACHE:-"$SHIFT_CACHE"}
export TEACHER_CACHE=${TEACHER_CACHE:-"$CACHE"}
export BUILD_SHIFT_CACHE=${BUILD_SHIFT_CACHE:-1}
export OUT=${OUT:-"$WORK_ROOT/train_shifted_visual_layout_adapter_hazard_teacher_v35"}

export EXPORT_OVERWRITE=${EXPORT_OVERWRITE:-1}
export INDEX_OVERWRITE=${INDEX_OVERWRITE:-1}
export OVERWRITE=${OVERWRITE:-0}

export BOOTSTRAP_ROOT=${BOOTSTRAP_ROOT:-"$HOME/teach2drive/workspace/teach2drive_bootstrap"}
export GARAGE_ROOT=${GARAGE_ROOT:-"$HOME/teach2drive/workspace/carla_garage"}
export TEAM_CONFIG=${TEAM_CONFIG:-"$HOME/teach2drive/checkpoints/transfuserpp/pretrained_models/all_towns"}

export EPOCHS=${EPOCHS:-45}
export BATCH_SIZE=${BATCH_SIZE:-128}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
export NUM_WORKERS=${NUM_WORKERS:-8}
export CACHE_WORKERS=${CACHE_WORKERS:-24}
export LR=${LR:-7e-5}
export VAL_RATIO=${VAL_RATIO:-0.15}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export SKIP_INVALID_MOTION=${SKIP_INVALID_MOTION:-1}

export MOVING_SPEED_THRESHOLD=${MOVING_SPEED_THRESHOLD:-1.0}
export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.0}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-1.0}

# Keep route geometry very close to the teacher. For speed, follow the teacher
# on normal frames but switch to expert targets on labeled hazards.
export TEACHER_TARGET_BLEND=${TEACHER_TARGET_BLEND:-0.0}
export TEACHER_TRAJ_BLEND=${TEACHER_TRAJ_BLEND:-0.95}
export TEACHER_SPEED_TARGET_BLEND=${TEACHER_SPEED_TARGET_BLEND:-0.90}
export TEACHER_STOP_TARGET_BLEND=${TEACHER_STOP_TARGET_BLEND:-0.0}
export HAZARD_STOP_REASONS=${HAZARD_STOP_REASONS:-"traffic_light,stop_sign,front_vehicle,junction_yield"}
export HAZARD_SPEED_TARGET_BLEND=${HAZARD_SPEED_TARGET_BLEND:-0.0}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-1.5}

export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-1.0}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.05}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.70}

export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.0}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.0}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.0}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-0.3}
export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.0}

# Slightly stronger path/speed shape regularization to reduce S-shaped tracking.
export SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.04}
export SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.02}
export TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.05}
export TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.02}

export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.01}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.03}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.05}
export STOP_POSITIVE_LOSS_SCALE=${STOP_POSITIVE_LOSS_SCALE:-0.75}
export STOP_NEGATIVE_LOSS_SCALE=${STOP_NEGATIVE_LOSS_SCALE:-1.25}
export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-4}

exec bash configs/train_tfpp_shifted_adapter_server.sh
