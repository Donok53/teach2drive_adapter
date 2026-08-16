#!/usr/bin/env bash
set -euo pipefail

# Exact-input TF++ adapter training on Tesla/shifted-rig data.
# By default this is canonical-free. Setting TEACHER_VIEW_ROOT enables the paired
# canonical-view feature teacher used by the best historical v4 recipe.

RUN_NAME=${RUN_NAME:-train_exact_target_only}
RUN_DIR=${RUN_DIR:-/data/dataset/byeongjae/runs/${RUN_NAME}}
OUT_DIR="${RUN_DIR}/train"
DATA_ROOT=${DATA_ROOT:-/data/dataset/byeongjae}
CODE_ROOT=${CODE_ROOT:-/data/users/byeongjae/code}
INDEX=${INDEX:-${DATA_ROOT}/datasets/t2d_gist3h_index.npz}
EPISODE_ROOT=${EPISODE_ROOT:-${DATA_ROOT}/datasets/t2d_pdm_lite_front_triplet_shifted_3h_tfpp_exact}
MEASUREMENT_ROOT=${MEASUREMENT_ROOT:-}
FEATURE_DRIFT_WEIGHT=${FEATURE_DRIFT_WEIGHT:-0.0}
BATCH_SIZE=${BATCH_SIZE:-12}
NUM_WORKERS=${NUM_WORKERS:-4}
EPOCHS=${EPOCHS:-20}
SEED=${SEED:-91}
LR=${LR:-1e-05}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-8}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16.0}
STAGE_ADAPTER_LAYERS=${STAGE_ADAPTER_LAYERS:-all}
STAGE_ADAPTER_MODALITIES=${STAGE_ADAPTER_MODALITIES:-all}
DISABLE_FUSION_ADAPTER=${DISABLE_FUSION_ADAPTER:-0}
EXTRINSIC_AWARE=${EXTRINSIC_AWARE:-1}
SOURCE_PROFILE=${SOURCE_PROFILE:-front_triplet_shifted}
TEACHER_VIEW_AS_INPUT=${TEACHER_VIEW_AS_INPUT:-0}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}
FREEZE_INIT_AS_TEACHER=${FREEZE_INIT_AS_TEACHER:-0}
FREEZE_ADAPTER_INCLUDE=${FREEZE_ADAPTER_INCLUDE:-}
HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-0}
BLOCKS=${BLOCKS:-2}
PDM_BEHAVIOR_LOSS_WEIGHT=${PDM_BEHAVIOR_LOSS_WEIGHT:-0.0}
PDM_LATERAL_LOSS_WEIGHT=${PDM_LATERAL_LOSS_WEIGHT:-0.0}
PDM_CONTROLLER_LOSS_WEIGHT=${PDM_CONTROLLER_LOSS_WEIGHT:-0.0}
PDM_PLAN_STEER_LOSS_WEIGHT=${PDM_PLAN_STEER_LOSS_WEIGHT:-0.0}
PDM_PLAN_THROTTLE_LOSS_WEIGHT=${PDM_PLAN_THROTTLE_LOSS_WEIGHT:-0.0}
PDM_PLAN_BRAKE_LOSS_WEIGHT=${PDM_PLAN_BRAKE_LOSS_WEIGHT:-0.0}
OUTPUT_PRIOR_XY_LOSS_WEIGHT=${OUTPUT_PRIOR_XY_LOSS_WEIGHT:-0.0}
OUTPUT_PRIOR_SPEED_LOSS_WEIGHT=${OUTPUT_PRIOR_SPEED_LOSS_WEIGHT:-0.0}
CONTROL_LOSS_WEIGHT=${CONTROL_LOSS_WEIGHT:-0.9}
STOP_STATE_AUX_LOSS_WEIGHT=${STOP_STATE_AUX_LOSS_WEIGHT:-0.15}
STOP_REASON_AUX_LOSS_WEIGHT=${STOP_REASON_AUX_LOSS_WEIGHT:-0.1}
OUTPUT_RESIDUAL=${OUTPUT_RESIDUAL:-0}
OUTPUT_RESIDUAL_HIDDEN_DIM=${OUTPUT_RESIDUAL_HIDDEN_DIM:-256}
OUTPUT_RESIDUAL_CHECKPOINT_SCALE=${OUTPUT_RESIDUAL_CHECKPOINT_SCALE:-0.75}
OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE=${OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE:-1.5}
OUTPUT_RESIDUAL_GATE_BIAS=${OUTPUT_RESIDUAL_GATE_BIAS:--2.0}
OUTPUT_RESIDUAL_DROPOUT=${OUTPUT_RESIDUAL_DROPOUT:-0.0}
OUTPUT_RESIDUAL_CHECKPOINT_LATERAL_ONLY=${OUTPUT_RESIDUAL_CHECKPOINT_LATERAL_ONLY:-0}
FREEZE_OUTPUT_RESIDUAL=${FREEZE_OUTPUT_RESIDUAL:-0}
TEACHER_DISTILL_ONLY=${TEACHER_DISTILL_ONLY:-0}
TEACHER_CHECKPOINT_LOSS_WEIGHT=${TEACHER_CHECKPOINT_LOSS_WEIGHT:-1.0}
ROUTE_TARGET_ONLY=${ROUTE_TARGET_ONLY:-0}
TFPP_ORIGINAL_OBJECTIVE=${TFPP_ORIGINAL_OBJECTIVE:-0}
TFPP_CHECKPOINT_LOSS_WEIGHT=${TFPP_CHECKPOINT_LOSS_WEIGHT:-1.0}
TFPP_TARGET_SPEED_LOSS_WEIGHT=${TFPP_TARGET_SPEED_LOSS_WEIGHT:-1.0}
ROUTE_TARGET_LEN=${ROUTE_TARGET_LEN:-10}
ROUTE_TARGET_SOURCE=${ROUTE_TARGET_SOURCE:-measurement_route}
FUTURE_EGO_MAX_HORIZON_S=${FUTURE_EGO_MAX_HORIZON_S:-6.0}
ROUTE_LATERAL_LOSS_WEIGHT=${ROUTE_LATERAL_LOSS_WEIGHT:-1.0}
ROUTE_HEADING_LOSS_WEIGHT=${ROUTE_HEADING_LOSS_WEIGHT:-0.25}
ROUTE_AIM_ANGLE_LOSS_WEIGHT=${ROUTE_AIM_ANGLE_LOSS_WEIGHT:-0.0}
ROUTE_STRAIGHT_IDENTITY_LOSS_WEIGHT=${ROUTE_STRAIGHT_IDENTITY_LOSS_WEIGHT:-0.2}
ROUTE_TURN_GATE_LOSS_WEIGHT=${ROUTE_TURN_GATE_LOSS_WEIGHT:-0.1}
ROUTE_TURN_LATERAL_THRESHOLD_M=${ROUTE_TURN_LATERAL_THRESHOLD_M:-0.5}
ROUTE_TURN_ANGLE_THRESHOLD_RAD=${ROUTE_TURN_ANGLE_THRESHOLD_RAD:-0.08}
SELECTION_METRIC=${SELECTION_METRIC:-loss}
LORA_INCLUDE=${LORA_INCLUDE:-^join\.,^checkpoint_decoder\.(encoder|decoder)\.,^target_speed_network\.}
LORA_REQUIRE_MODULES=${LORA_REQUIRE_MODULES:-}
UNFREEZE_INCLUDE=${UNFREEZE_INCLUDE:-}
TEACHER_VIEW_ROOT=${TEACHER_VIEW_ROOT:-}
TEACHER_VIEW_DIRNAME=${TEACHER_VIEW_DIRNAME:-rgb_canonical}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-0}

EXTRA_ARGS=()
if [[ -n "${UNFREEZE_INCLUDE}" ]]; then
  EXTRA_ARGS+=(--unfreeze-include "${UNFREEZE_INCLUDE}")
fi
if [[ -n "${TEACHER_VIEW_ROOT}" ]]; then
  EXTRA_ARGS+=(--teacher-view-root "${TEACHER_VIEW_ROOT}" --teacher-view-dirname "${TEACHER_VIEW_DIRNAME}")
fi
if [[ "${DISABLE_FUSION_ADAPTER}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-fusion-adapter)
fi
if [[ "${EXTRINSIC_AWARE}" == "1" ]]; then
  EXTRA_ARGS+=(--extrinsic-aware)
fi
if [[ "${TEACHER_VIEW_AS_INPUT}" == "1" ]]; then
  EXTRA_ARGS+=(--teacher-view-as-input)
fi
if [[ "${FREEZE_INIT_AS_TEACHER}" == "1" ]]; then
  EXTRA_ARGS+=(--freeze-init-as-teacher)
fi
if [[ -n "${FREEZE_ADAPTER_INCLUDE}" ]]; then
  EXTRA_ARGS+=(--freeze-adapter-include "${FREEZE_ADAPTER_INCLUDE}")
fi
if [[ "${OUTPUT_RESIDUAL}" == "1" ]]; then
  EXTRA_ARGS+=(--output-residual)
fi
if [[ -n "${MEASUREMENT_ROOT}" ]]; then
  EXTRA_ARGS+=(--measurement-root "${MEASUREMENT_ROOT}")
fi
if [[ "${OUTPUT_RESIDUAL_CHECKPOINT_LATERAL_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--output-residual-checkpoint-lateral-only)
fi
if [[ "${FREEZE_OUTPUT_RESIDUAL}" == "1" ]]; then
  EXTRA_ARGS+=(--freeze-output-residual)
fi
if [[ "${TEACHER_DISTILL_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--teacher-distill-only)
fi
if [[ "${ROUTE_TARGET_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--route-target-only)
fi
if [[ "${TFPP_ORIGINAL_OBJECTIVE}" == "1" ]]; then
  EXTRA_ARGS+=(--tfpp-original-objective)
fi

mkdir -p "${OUT_DIR}"
cd "${CODE_ROOT}/teach2drive_adapter"

echo "=== exact target-only adapter start $(date '+%F %T') ==="
echo "run_name=${RUN_NAME} run_dir=${RUN_DIR} drift=${FEATURE_DRIFT_WEIGHT} batch=${BATCH_SIZE} seed=${SEED}"
echo "index=${INDEX} episode_root=${EPISODE_ROOT} teacher=${TEACHER_VIEW_ROOT:-none} measurement=${MEASUREMENT_ROOT:-none} selection=${SELECTION_METRIC} unfreeze=${UNFREEZE_INCLUDE:-none}"
echo "commit=$(git rev-parse --short HEAD 2>/dev/null || true)"

PYTHONPATH="${CODE_ROOT}/teach2drive_adapter:${CODE_ROOT}/carla_garage/team_code:${PYTHONPATH:-}" \
python -m teach2drive_adapter.train_transfuserpp_task_feature_adapter \
  --index "${INDEX}" \
  --episode-root-override "${EPISODE_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --garage-root "${CODE_ROOT}/carla_garage" \
  --team-config "${DATA_ROOT}/checkpoints/transfuserpp/pretrained_models/all_towns" \
  --checkpoint '' \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --cameras front \
  --tfpp-camera front \
  --command-mode target_angle \
  --image-size 1024 512 \
  --lidar-size 256 \
  --source-profile "${SOURCE_PROFILE}" \
  --extrinsic-hidden-dim 64 \
  --extrinsic-dropout 0.0 \
  --hidden-channels "${HIDDEN_CHANNELS}" \
  --blocks "${BLOCKS}" \
  --dropout 0.0 \
  --stage-adapter-layers "${STAGE_ADAPTER_LAYERS}" \
  --stage-adapter-modalities "${STAGE_ADAPTER_MODALITIES}" \
  --stage-feature-adapter-blend 1.0 \
  --fusion-adapter-blend 1.0 \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --lora-dropout 0.02 \
  --lora-include "${LORA_INCLUDE}" \
  --lora-exclude '' \
  --lora-require-modules "${LORA_REQUIRE_MODULES}" \
  --epochs "${EPOCHS}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE}" \
  --early-stop-min-delta 0.0 \
  --selection-metric "${SELECTION_METRIC}" \
  --selection-mode min \
  --save-epoch-checkpoints \
  --epoch-checkpoint-dir epoch_checkpoints \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --val-ratio 0.15 \
  --seed "${SEED}" \
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
  --feature-drift-loss-weight "${FEATURE_DRIFT_WEIGHT}" \
  --output-prior-xy-loss-weight "${OUTPUT_PRIOR_XY_LOSS_WEIGHT}" \
  --output-prior-speed-loss-weight "${OUTPUT_PRIOR_SPEED_LOSS_WEIGHT}" \
  --aux-hidden-dim 256 \
  --control-loss-weight "${CONTROL_LOSS_WEIGHT}" \
  --pdm-behavior-loss-weight "${PDM_BEHAVIOR_LOSS_WEIGHT}" \
  --pdm-lateral-loss-weight "${PDM_LATERAL_LOSS_WEIGHT}" \
  --pdm-controller-loss-weight "${PDM_CONTROLLER_LOSS_WEIGHT}" \
  --pdm-plan-steer-loss-weight "${PDM_PLAN_STEER_LOSS_WEIGHT}" \
  --pdm-plan-throttle-loss-weight "${PDM_PLAN_THROTTLE_LOSS_WEIGHT}" \
  --pdm-plan-brake-loss-weight "${PDM_PLAN_BRAKE_LOSS_WEIGHT}" \
  --output-residual-hidden-dim "${OUTPUT_RESIDUAL_HIDDEN_DIM}" \
  --output-residual-checkpoint-scale "${OUTPUT_RESIDUAL_CHECKPOINT_SCALE}" \
  --output-residual-speed-logit-scale "${OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE}" \
  --output-residual-gate-bias "${OUTPUT_RESIDUAL_GATE_BIAS}" \
  --output-residual-dropout "${OUTPUT_RESIDUAL_DROPOUT}" \
  --teacher-checkpoint-loss-weight "${TEACHER_CHECKPOINT_LOSS_WEIGHT}" \
  --route-target-len "${ROUTE_TARGET_LEN}" \
  --route-target-source "${ROUTE_TARGET_SOURCE}" \
  --tfpp-checkpoint-loss-weight "${TFPP_CHECKPOINT_LOSS_WEIGHT}" \
  --tfpp-target-speed-loss-weight "${TFPP_TARGET_SPEED_LOSS_WEIGHT}" \
  --future-ego-max-horizon-s "${FUTURE_EGO_MAX_HORIZON_S}" \
  --route-lateral-loss-weight "${ROUTE_LATERAL_LOSS_WEIGHT}" \
  --route-heading-loss-weight "${ROUTE_HEADING_LOSS_WEIGHT}" \
  --route-aim-angle-loss-weight "${ROUTE_AIM_ANGLE_LOSS_WEIGHT}" \
  --route-straight-identity-loss-weight "${ROUTE_STRAIGHT_IDENTITY_LOSS_WEIGHT}" \
  --route-turn-gate-loss-weight "${ROUTE_TURN_GATE_LOSS_WEIGHT}" \
  --route-turn-lateral-threshold-m "${ROUTE_TURN_LATERAL_THRESHOLD_M}" \
  --route-turn-angle-threshold-rad "${ROUTE_TURN_ANGLE_THRESHOLD_RAD}" \
  --stop-state-aux-loss-weight "${STOP_STATE_AUX_LOSS_WEIGHT}" \
  --stop-reason-aux-loss-weight "${STOP_REASON_AUX_LOSS_WEIGHT}" \
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
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --max-val-samples "${MAX_VAL_SAMPLES}" \
  --step-log-every 50 \
  "${EXTRA_ARGS[@]}"

echo "=== exact target-only adapter finished $(date '+%F %T') ==="
