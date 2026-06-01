#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-/home/jovyan/teach2drive/logs}"
RUN_ROOT="${RUN_ROOT:-/home/jovyan/dataset/byeongjae/runs}"
PY="${PY:-/tmp/pytorch}"
BATCH_SIZE="${BATCH_SIZE:-24}"
SELECTION_METRIC="${SELECTION_METRIC:-closed_loop_proxy}"
SELECTION_MODE="${SELECTION_MODE:-min}"
OVERWRITE="${OVERWRITE:-1}"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

launch_one() {
  local name="$1"
  local gpu="$2"
  local config="$3"
  local out="${RUN_ROOT}/${name}"
  local log="${LOG_DIR}/${name}.log"
  local pidfile="${LOG_DIR}/${name}.pid"

  echo "start ${name} gpu=${gpu} out=${out} log=${log}"
  setsid env \
    PY="${PY}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OUT="${out}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    SELECTION_METRIC="${SELECTION_METRIC}" \
    SELECTION_MODE="${SELECTION_MODE}" \
    OVERWRITE="${OVERWRITE}" \
    bash "${ROOT}/${config}" \
    > "${log}" 2>&1 < /dev/null &
  echo "$!" > "${pidfile}"
}

launch_one cl1_b24 0 configs/train_tfpp_tesla_town13_expert_only_lora8_blend1_server.sh
launch_one cl2_b24 1 configs/train_tfpp_tesla_town13_expert_only_lora16_head_blend1_server.sh
launch_one cl3_b24 2 configs/train_tfpp_tesla_town13_expert_stop_aux_lora8_blend1_server.sh
launch_one cl4_b24 3 configs/train_tfpp_tesla_town13_expert_control_aux_lora8_blend1_server.sh
