#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Feature+Fusion+Output Adapter v2: weak residual.
#
# v1 let the output adapter override the recovered Feature+Fusion policy too
# much. In closed loop it repeatedly predicted near-zero speed and non-trivial
# brake at launch, so this preset trains only a small residual correction:
#   - reuse the Feature+Fusion prior cache when available
#   - keep predictions close to the Feature+Fusion prior
#   - disable the direct control head
#   - avoid stop/traffic-light classifiers as behavior levers
#
# Evaluation should usually use PID control and a small speed blend, not direct
# longitudinal control.

export DATA_ROOT=${DATA_ROOT:-"$HOME/dataset/byeongjae/datasets/t2d_transfuserpp_paired_24ep_motion_safe"}
export WORK_ROOT=${WORK_ROOT:-"$HOME/teach2drive/runs/tfpp_feature_then_fusion_output_adapter_v2_weak_residual"}
export OUT=${OUT:-"$WORK_ROOT/train_front_triplet_shifted_feature_then_fusion_output_adapter_v2_weak_residual"}
export FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT=${FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT:-"$HOME/dataset/byeongjae/runs/tfpp_feature_then_fusion_adapter_v1/train_feature_then_fusion_adapter_v1/best_model.pt"}

export PREV_WORK_ROOT=${PREV_WORK_ROOT:-"$HOME/dataset/byeongjae/runs/tfpp_feature_then_fusion_output_adapter_v1"}
if [[ -z "${CACHE:-}" && -f "$PREV_WORK_ROOT/cache/front_triplet_shifted_prior_cache.npz" ]]; then
  export CACHE="$PREV_WORK_ROOT/cache/front_triplet_shifted_prior_cache.npz"
fi

export CAMERAS=${CAMERAS:-left,front,right}
export EPOCHS=${EPOCHS:-60}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-12}
export BATCH_SIZE=${BATCH_SIZE:-128}
export CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-128}
export DATA_PARALLEL=${DATA_PARALLEL:-1}
export CACHE_DATA_PARALLEL=${CACHE_DATA_PARALLEL:-0}
export TRAIN_DATA_PARALLEL=${TRAIN_DATA_PARALLEL:-$DATA_PARALLEL}
export OVERWRITE=${OVERWRITE:-0}

export STAGE_FEATURE_ADAPTER_BLEND=${STAGE_FEATURE_ADAPTER_BLEND:-1.0}
export FUSION_ADAPTER_BLEND=${FUSION_ADAPTER_BLEND:-1.0}

export HIDDEN_DIM=${HIDDEN_DIM:-512}
export LAYOUT_HIDDEN_DIM=${LAYOUT_HIDDEN_DIM:-128}
export VISUAL_DIM=${VISUAL_DIM:-256}
export VISUAL_TOKEN_DIM=${VISUAL_TOKEN_DIM:-192}
export VISUAL_LAYERS=${VISUAL_LAYERS:-2}
export VISUAL_HEADS=${VISUAL_HEADS:-4}
export LR=${LR:-5e-5}
export WEIGHT_DECAY=${WEIGHT_DECAY:-2e-4}

export CAMERA_DROPOUT_PROB=${CAMERA_DROPOUT_PROB:-0.0}
export FRONT_CAMERA_DROPOUT_PROB=${FRONT_CAMERA_DROPOUT_PROB:-0.0}

export MOVING_SAMPLE_WEIGHT=${MOVING_SAMPLE_WEIGHT:-1.0}
export STOPPED_SAMPLE_WEIGHT=${STOPPED_SAMPLE_WEIGHT:-1.0}
export HAZARD_SAMPLE_WEIGHT=${HAZARD_SAMPLE_WEIGHT:-1.0}

export XY_LOSS_WEIGHT=${XY_LOSS_WEIGHT:-0.15}
export YAW_LOSS_WEIGHT=${YAW_LOSS_WEIGHT:-0.01}
export SPEED_LOSS_WEIGHT=${SPEED_LOSS_WEIGHT:-0.12}

# Speed supervision is mostly the Feature+Fusion prior, with a small pull
# toward expert labels. This prevents the adapter from inventing stop behavior.
export SPEED_TEACHER_BLEND=${SPEED_TEACHER_BLEND:-0.75}
export SPEED_DISTILL_LOSS_WEIGHT=${SPEED_DISTILL_LOSS_WEIGHT:-0.50}

export SPEED_FLOOR_MASK=${SPEED_FLOOR_MASK:-target}
export SPEED_FLOOR_TARGET_THRESHOLD=${SPEED_FLOOR_TARGET_THRESHOLD:-2.0}
export SPEED_FLOOR_LOSS_WEIGHT=${SPEED_FLOOR_LOSS_WEIGHT:-0.02}
export SPEED_FLOOR_MPS=${SPEED_FLOOR_MPS:-0.8}

export LAUNCH_CURRENT_SPEED_THRESHOLD=${LAUNCH_CURRENT_SPEED_THRESHOLD:-0.8}
export LAUNCH_TARGET_SPEED_THRESHOLD=${LAUNCH_TARGET_SPEED_THRESHOLD:-2.0}
export LAUNCH_SAMPLE_WEIGHT=${LAUNCH_SAMPLE_WEIGHT:-1.0}
export LAUNCH_SPEED_FLOOR_LOSS_WEIGHT=${LAUNCH_SPEED_FLOOR_LOSS_WEIGHT:-0.05}
export LAUNCH_SPEED_FLOOR_MPS=${LAUNCH_SPEED_FLOOR_MPS:-1.2}
export LAUNCH_STOP_NEGATIVE=${LAUNCH_STOP_NEGATIVE:-0}

export RELEASE_TARGET_SPEED_THRESHOLD=${RELEASE_TARGET_SPEED_THRESHOLD:-1.0}
export RELEASE_SAMPLE_WEIGHT=${RELEASE_SAMPLE_WEIGHT:-1.0}
export RELEASE_SPEED_FLOOR_LOSS_WEIGHT=${RELEASE_SPEED_FLOOR_LOSS_WEIGHT:-0.03}
export RELEASE_SPEED_FLOOR_MPS=${RELEASE_SPEED_FLOOR_MPS:-1.2}
export RELEASE_STOP_NEGATIVE=${RELEASE_STOP_NEGATIVE:-0}

export SPEED_DELTA_LOSS_WEIGHT=${SPEED_DELTA_LOSS_WEIGHT:-0.02}
export SPEED_CURVATURE_LOSS_WEIGHT=${SPEED_CURVATURE_LOSS_WEIGHT:-0.01}
export TRAJ_DELTA_LOSS_WEIGHT=${TRAJ_DELTA_LOSS_WEIGHT:-0.02}
export TRAJ_CURVATURE_LOSS_WEIGHT=${TRAJ_CURVATURE_LOSS_WEIGHT:-0.01}

# This is the main knob: keep the output residual close to the recovered
# Feature+Fusion prior.
export PRIOR_LOSS_WEIGHT=${PRIOR_LOSS_WEIGHT:-2.0}

# Disable direct throttle/brake learning for this weak-residual run. Closed-loop
# evaluation should let TransFuser++ PID consume the slightly adapted target.
export CONTROL_LOSS_WEIGHT=${CONTROL_LOSS_WEIGHT:-0.0}
export CONTROL_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT=${CONTROL_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT:-0.0}
export CONTROL_LAUNCH_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT=${CONTROL_LAUNCH_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT:-0.0}
export CONTROL_LAUNCH_THROTTLE_FLOOR_LOSS_WEIGHT=${CONTROL_LAUNCH_THROTTLE_FLOOR_LOSS_WEIGHT:-0.0}
export CONTROL_RELEASE_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT=${CONTROL_RELEASE_BRAKE_FALSE_POSITIVE_LOSS_WEIGHT:-0.0}
export CONTROL_RELEASE_THROTTLE_FLOOR_LOSS_WEIGHT=${CONTROL_RELEASE_THROTTLE_FLOOR_LOSS_WEIGHT:-0.0}
export CONTROL_BRAKE_TRUE_POSITIVE_LOSS_WEIGHT=${CONTROL_BRAKE_TRUE_POSITIVE_LOSS_WEIGHT:-0.0}
export CONTROL_HAZARD_BRAKE_TRUE_POSITIVE_LOSS_WEIGHT=${CONTROL_HAZARD_BRAKE_TRUE_POSITIVE_LOSS_WEIGHT:-0.0}

export STOP_SPEED_CEILING_LOSS_WEIGHT=${STOP_SPEED_CEILING_LOSS_WEIGHT:-0.0}
export HAZARD_STOP_SPEED_CEILING_LOSS_WEIGHT=${HAZARD_STOP_SPEED_CEILING_LOSS_WEIGHT:-0.0}

export STOP_LOSS_AFTER_EPOCH=${STOP_LOSS_AFTER_EPOCH:-9999}
export STOP_LOSS_WEIGHT=${STOP_LOSS_WEIGHT:-0.0}
export STOP_STATE_LOSS_WEIGHT=${STOP_STATE_LOSS_WEIGHT:-0.0}
export STOP_REASON_LOSS_WEIGHT=${STOP_REASON_LOSS_WEIGHT:-0.0}

exec bash configs/train_tfpp_vehicle_b_expert_v1_server.sh
