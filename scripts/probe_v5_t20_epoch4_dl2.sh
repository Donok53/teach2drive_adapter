#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/media/aimlab/HDD00/users/byeongjae/workspace/teach2drive}
REPO=${REPO:-${ROOT}/code/teach2drive_adapter}
DATA=${DATA:-${ROOT}/data}
TRAIN_RUN=${TRAIN_RUN:-train_stage1a_speed_runtime_v5_t20_gpu1_20260818}
TRAIN_SESSION=${TRAIN_SESSION:-stage1a_v5_t20_gpu1}
PROBE_EPOCH=${PROBE_EPOCH:-004}
CHECKPOINT_HOST=${CHECKPOINT_HOST:-${DATA}/runs/${TRAIN_RUN}/train/epoch_checkpoints/epoch_${PROBE_EPOCH}.pt}
CHECKPOINT=${CHECKPOINT:-/data/dataset/byeongjae/runs/${TRAIN_RUN}/train/epoch_checkpoints/epoch_${PROBE_EPOCH}.pt}
RUN_NAME=${RUN_NAME:-probe_v5_t20_e4_preserve_m5_m11_20260818}
PROBE_TOTAL=${PROBE_TOTAL:-3}
PROBE_INDICES=${PROBE_INDICES:-0,4,10}
OUTPUT_RESIDUAL_HARD_GATE_THRESHOLD=${OUTPUT_RESIDUAL_HARD_GATE_THRESHOLD:-}
RECORD_VIDEO=${RECORD_VIDEO:-0}
VIDEO_RECORD_INDICES=${VIDEO_RECORD_INDICES:-}
# An explicitly empty value means evaluation should run concurrently without
# pausing training (for example, training GPU0 while probing on GPU1).
PAUSE_RUNS=${PAUSE_RUNS-train_stage1a_speed_runtime_v5_t10_gpu0_20260818,train_stage1a_speed_runtime_v5_t20_gpu1_20260818}
PROBE_MISSION_TIMEOUT=${PROBE_MISSION_TIMEOUT:-900}
PROBE_GPU=${PROBE_GPU:-0}
PROBE_PORT=${PROBE_PORT:-2080}
PROBE_TM_PORT=${PROBE_TM_PORT:-8080}
PROBE_AGENT_PATH=${PROBE_AGENT_PATH:-/data/users/byeongjae/code/teach2drive_adapter/scripts/tfpp_feature_then_fusion_peft_adapter_sensor_rig_agent.py}
RUN_ROOT=${DATA}/runs/${RUN_NAME}
LOG=${RUN_ROOT}/probe_controller.log
PAUSED_FILE=${RUN_ROOT}/paused_containers.txt

mkdir -p "${RUN_ROOT}"
: > "${PAUSED_FILE}"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${LOG}"
}

resume_training() {
  if [[ -s "${PAUSED_FILE}" ]]; then
    while read -r container; do
      [[ -n "${container}" ]] || continue
      docker unpause "${container}" >/dev/null 2>&1 || true
    done < "${PAUSED_FILE}"
    : > "${PAUSED_FILE}"
    log 'training containers resumed'
  fi
}
trap resume_training EXIT INT TERM

log "waiting for ${CHECKPOINT_HOST}"
while [[ ! -s "${CHECKPOINT_HOST}" ]]; do
  if ! tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    log "ERROR ${TRAIN_SESSION} ended before epoch ${PROBE_EPOCH} checkpoint"
    exit 3
  fi
  sleep 30
done
log "epoch ${PROBE_EPOCH} checkpoint ready"

# All DL2 closed-loop evaluations share CARLA port 2080. Serialize probes so
# two epoch watchers cannot start competing servers at the same time.
EVAL_LOCK=${EVAL_LOCK:-${DATA}/runs/.dl2_eval.lock}
exec 9>"${EVAL_LOCK}"
log "waiting for DL2 evaluation lock ${EVAL_LOCK}"
flock 9
log 'DL2 evaluation lock acquired'

# Pause only the two explicitly named training containers. Other user containers are
# never touched.  GPU memory remains allocated, but both A6000s have ample room
# for the evaluation and no training kernels contend with CARLA.
for container in $(docker ps -q --filter ancestor=teach2drive-adapter:dl2); do
  env_dump=$(docker inspect "${container}" --format '{{range .Config.Env}}{{println .}}{{end}}')
  container_run=$(sed -n 's/^RUN_NAME=//p' <<< "${env_dump}" | head -n 1)
  if [[ ",${PAUSE_RUNS}," == *",${container_run},"* ]]; then
    docker pause "${container}" >/dev/null
    printf '%s\n' "${container}" >> "${PAUSED_FILE}"
  fi
done
paused=$(wc -l < "${PAUSED_FILE}")
expected_paused=$(tr ',' '\n' <<< "${PAUSE_RUNS}" | sed '/^[[:space:]]*$/d' | sort -u | wc -l)
if [[ "${paused}" -ne "${expected_paused}" ]]; then
  log "ERROR expected ${expected_paused} paused training containers, found ${paused}"
  exit 4
fi
log "paused ${paused} selected training container(s)"

for _ in $(seq 1 60); do
  if ! timeout 1 bash -c "</dev/tcp/127.0.0.1/${PROBE_PORT}" 2>/dev/null; then
    break
  fi
  sleep 2
done

set +e
env GPU="${PROBE_GPU}" PORT="${PROBE_PORT}" TM_PORT="${PROBE_TM_PORT}" SEED=0 \
  RUN_NAME="${RUN_NAME}" CHECKPOINT="${CHECKPOINT}" \
  AGENT_PATH="${PROBE_AGENT_PATH}" \
  EGO_VEHICLE_MODEL=vehicle.tesla.model3 \
  TFPP_MODEL_ORDER=model_0030_1.pth,model_0030_0.pth,model_0030_2.pth \
  TOTAL="${PROBE_TOTAL}" MAX_ATTEMPTS=8 MISSION_INDICES="${PROBE_INDICES}" \
  RECORD_VIDEO="${RECORD_VIDEO}" VIDEO_RECORD_INDICES="${VIDEO_RECORD_INDICES}" \
  DEBUG=0 MISSION_TIMEOUT_SEC="${PROBE_MISSION_TIMEOUT}" \
  OUTPUT_RESIDUAL_BLEND=1.0 \
  OUTPUT_RESIDUAL_HARD_GATE_THRESHOLD="${OUTPUT_RESIDUAL_HARD_GATE_THRESHOLD}" \
  OUTPUT_TRACE=1 \
  bash "${REPO}/scripts/supervise_policy_preserving_adapter_eval_dl2.sh" \
  >> "${LOG}" 2>&1
status=$?
set -e

summary=${RUN_ROOT}/summary.tsv
rows=0
passes=0
if [[ -f "${summary}" ]]; then
  rows=$(awk 'NR>1 {n++} END {print n+0}' "${summary}")
  passes=$(awk -F '\t' 'NR>1 && $3=="PASS" {n++} END {print n+0}' "${summary}")
fi
log "probe complete exit=${status} pass=${passes}/${rows}"
[[ "${rows}" -ge "${PROBE_TOTAL}" ]]
