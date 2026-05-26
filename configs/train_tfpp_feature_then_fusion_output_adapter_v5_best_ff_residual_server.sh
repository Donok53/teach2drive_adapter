#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Feature+Fusion+Output Adapter v5: best feature/fusion anchor + conservative expert residual.
#
# This preset is the first "performance improvement" branch after the v4 policy
# recovery run. It keeps the strongest closed-loop feature+fusion adapter fixed
# and trains only a small deployed output residual on expert data.
#
# Differences from v3_balanced_residual:
#   - anchor: extrinsic feature+fusion v2 conservative instead of plain FF v1
#   - cache: rebuilt under this WORK_ROOT so the prior matches the v2 anchor
#   - prior weight: stronger, to avoid full-policy drift seen in v4/full fine-tune
#   - expert pull: moderate speed/trajectory correction, direct control disabled
#
# Recommended closed-loop eval:
#   TFPP_ADAPTER_SPEED_MODE=blend TFPP_ADAPTER_SPEED_BLEND=0.25..0.35
#   TFPP_ADAPTER_USE_STOP=0 TFPP_ADAPTER_CONTROL_MODE=pid

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_feature_then_fusion_output_adapter_v5_best_ff_residual"}
export OUT=${OUT:-"$WORK_ROOT/train_front_triplet_shifted_feature_then_fusion_output_adapter_v5_best_ff_residual"}
export FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT=${FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT:-"$HOME/dataset/byeongjae/runs/tfpp_extrinsic_feature_then_fusion_adapter_v2_conservative/train_extrinsic_feature_then_fusion_adapter_v2_conservative/best_model.pt"}

export PROFILE=${PROFILE:-front_triplet_shifted}
export CAMERAS=${CAMERAS:-left,front,right}
export EPOCHS=${EPOCHS:-60}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-10}
export BATCH_SIZE=${BATCH_SIZE:-128}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
export NUM_WORKERS=${NUM_WORKERS:-8}
export CACHE_WORKERS=${CACHE_WORKERS:-16}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export CACHE_DATA_PARALLEL=${CACHE_DATA_PARALLEL:-$DATA_PARALLEL}
export TRAIN_DATA_PARALLEL=${TRAIN_DATA_PARALLEL:-$DATA_PARALLEL}
export OVERWRITE=${OVERWRITE:-0}

export STAGE_FEATURE_ADAPTER_BLEND=${STAGE_FEATURE_ADAPTER_BLEND:-1.0}
export FUSION_ADAPTER_BLEND=${FUSION_ADAPTER_BLEND:-1.0}

# Keep the residual compact. The base feature/fusion adapter is already doing
# the sensor-domain correction; this head should only nudge policy outputs.
export HIDDEN_DIM=${HIDDEN_DIM:-512}
export LAYOUT_HIDDEN_DIM=${LAYOUT_HIDDEN_DIM:-128}
export VISUAL_DIM=${VISUAL_DIM:-256}
export VISUAL_TOKEN_DIM=${VISUAL_TOKEN_DIM:-192}
export VISUAL_LAYERS=${VISUAL_LAYERS:-2}
export VISUAL_HEADS=${VISUAL_HEADS:-4}
export LR=${LR:-6e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-2e-4}

export CAMERA_DROPOUT_PROB=${CAMERA_DROPOUT_PROB:-0.0}
export FRONT_CAMERA_DROPOUT_PROB=${FRONT_CAMERA_DROPOUT_PROB:-0.0}

export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.05}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-1.00}
export HAZARD_STOP_REASONS=${HAZARD_STOP_REASONS:-"traffic_light,stop_sign,front_vehicle,junction_yield"}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-1.30}

export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-0.14}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.01}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.28}

# Blend speed target toward the anchored FF policy. The expert still teaches
# release/side-vehicle behavior, but the anchor prevents slow/stuck collapse.
export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.35}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.25}
export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-0.90}

export SPEED_FLOOR_MASK=${SPEED_FLOOR_MASK:-target}
export SPEED_FLOOR_TARGET_THRESHOLD=${SPEED_FLOOR_TARGET_THRESHOLD:-2.0}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.035}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-0.9}

export LAUNCH_CURRENT_SPEED_THRESHOLD=${LAUNCH_CURRENT_SPEED_THRESHOLD:-0.8}
export LAUNCH_TARGET_SPEED_THRESHOLD=${LAUNCH_TARGET_SPEED_THRESHOLD:-2.0}
export LAUNCH_SAMPLE_WEIGHT=${LAUNCH_SAMPLE_WEIGHT:-1.10}
export LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=${LAUNCH_SPEED_FLOOR_LOSS_WEIGHT:-0.07}
export LAUNCH_SPEED_FLOOR_MPS=${LAUNCH_SPEED_FLOOR_MPS:-1.4}
export LAUNCH_STOP_NEGATIVE=${LAUNCH_STOP_NEGATIVE:-0}

export RELEASE_TARGET_SPEED_THRESHOLD=${RELEASE_TARGET_SPEED_THRESHOLD:-1.0}
export RELEASE_SAMPLE_WEIGHT=${RELEASE_SAMPLE_WEIGHT:-1.10}
export RELEASE_SPEED_FLOOR_LOSS_WEIGHT=${RELEASE_SPEED_FLOOR_LOSS_WEIGHT:-0.05}
export RELEASE_SPEED_FLOOR_MPS=${RELEASE_SPEED_FLOOR_MPS:-1.4}
export RELEASE_STOP_NEGATIVE=${RELEASE_STOP_NEGATIVE:-0}

export SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.03}
export SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.01}
export TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.02}
export TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.01}

# Soft ceiling only; do not train/deploy explicit stop heads for this first pass.
export STOP_SPEED_TARGET_THRESHOLD=${STOP_SPEED_TARGET_THRESHOLD:-2.0}
export STOP_SPEED_CEILING_MPS=${STOP_SPEED_CEILING_MPS:-0.8}
export STOP_SPEED_CEILING_LOSS_WEIGHT=${STOP_SPEED_CEILING_LOSS_WEIGHT:-0.10}
export HAZARD_STOP_SPEED_CEILING_LOSS_WEIGHT=${HAZARD_STOP_SPEED_CEILING_LOSS_WEIGHT:-0.14}

export CONTROL_LOSS_WEIGHT=${CONTROL_LOSS_WEIGHT:-0.0}
export CONTROL_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT=${CONTROL_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT:-0.0}
export CONTROL_LAUNCH_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT=${CONTROL_LAUNCH_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT:-0.0}
export CONTROL_LAUNCH_THROTTLE_FLOOR_LOSS_WEIGHT=${CONTROL_LAUNCH_THROTTLE_FLOOR_LOSS_WEIGHT:-0.0}
export CONTROL_RELEASE_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT=${CONTROL_RELEASE_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT:-0.0}
export CONTROL_RELEASE_THROTTLE_FLOOR_LOSS_WEIGHT=${CONTROL_RELEASE_THROTTLE_FLOOR_LOSS_WEIGHT:-0.0}
export CONTROL_BRAKE_TRUE_POSITIVE_LOSS_WEIGHT=${CONTROL_BRAKE_TRUE_POSITIVE_LOSS_WEIGHT:-0.0}
export CONTROL_HAZARD_BRAKE_TRUE_POSITIVE_LOSS_WEIGHT=${CONTROL_HAZARD_BRAKE_TRUE_POSITIVE_LOSS_WEIGHT:-0.0}

export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-9999}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.0}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.0}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.0}

exec bash configs/train_tfpp_vehicle_b_expert_v1_server.sh
