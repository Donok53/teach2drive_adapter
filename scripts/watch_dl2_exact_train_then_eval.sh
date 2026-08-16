#!/usr/bin/env bash
set -u

ADAPTER_ROOT=${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}
REMOTE=${REMOTE:-DL2}
REMOTE_RUN_ROOT=${REMOTE_RUN_ROOT:-/media/aimlab/HDD00/users/byeongjae/workspace/teach2drive/data/runs}
QUEUE_ROOT=${QUEUE_ROOT:-${ADAPTER_ROOT}/runs/dl2_exact_eval_queue_20260811}
POLL_SEC=${POLL_SEC:-300}
GPU=${GPU:-1}

mkdir -p "${QUEUE_ROOT}/checkpoints" "${QUEUE_ROOT}/state"
echo "$$" > "${QUEUE_ROOT}/state/watcher.pid"
exec >> "${QUEUE_ROOT}/watcher.log" 2>&1

declare -A REMOTE_RUN=(
  [v4_canonical]="train_exact_v4_canonical_gpu0"
  [v9_nocanonical]="train_exact_v9_nocanonical_gpu1"
)
declare -A REMOTE_TMUX=(
  [v4_canonical]="t2d_exact_v4_canonical_gpu0"
  [v9_nocanonical]="t2d_exact_v9_gpu1"
)

log() {
  echo "[$(date '+%F %T')] $*"
}

gpu_compute_pids() {
  nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d'
}

remote_status() {
  local key=$1
  local run=${REMOTE_RUN[$key]}
  local session=${REMOTE_TMUX[$key]}
  local stop_condition="false"
  local marker_name=""
  if [[ "${key}" == "v4_canonical" ]]; then
    stop_condition="grep -Eq '^epoch=011 .*new_best=0' '${REMOTE_RUN_ROOT}/${run}/launcher.log'"
    marker_name="STOPPED_AFTER_EPOCH11_NO_IMPROVEMENT"
  elif [[ "${key}" == "v9_nocanonical" ]]; then
    stop_condition="grep -Eq '^epoch=014 .*new_best=0' '${REMOTE_RUN_ROOT}/${run}/launcher.log'"
    marker_name="STOPPED_AFTER_EPOCH14_NO_IMPROVEMENT"
  fi
  ssh "${REMOTE}" "train='${REMOTE_RUN_ROOT}/${run}/train'; marker='${REMOTE_RUN_ROOT}/${run}/${marker_name}'; if test -s \"\$train/summary.json\" && test -s \"\$train/best_model.pt\"; then stat -c 'READY %Y' \"\$train/summary.json\"; elif test -n '${marker_name}' && test -s \"\$marker\" && test -s \"\$train/best_model.pt\"; then stat -c 'READY %Y' \"\$marker\"; elif ${stop_condition}; then echo STOP_REQUESTED; elif tmux has-session -t '${session}' 2>/dev/null; then echo RUNNING; else echo FAILED; fi"
}

stop_after_no_improvement() {
  local key=$1
  local epoch=$2
  local marker_name=$3
  local run=${REMOTE_RUN[$key]}
  local session=${REMOTE_TMUX[$key]}
  log "${key}: epoch ${epoch} did not improve; stopping training and keeping the existing best checkpoint"
  ssh "${REMOTE}" "
    set -u
    root='${REMOTE_RUN_ROOT}/${run}'
    line=\$(grep -E '^epoch=${epoch} .*new_best=0' \"\$root/launcher.log\" | tail -n 1)
    tmux send-keys -t '${session}' C-c 2>/dev/null || true
    sleep 5
    for id in \$(docker ps -q --filter ancestor=teach2drive-adapter:dl2); do
      if docker inspect -f '{{json .Config.Cmd}}' \"\$id\" | grep -q 'RUN_NAME=${run}'; then
        docker stop -t 20 \"\$id\" >/dev/null || true
      fi
    done
    tmux kill-session -t '${session}' 2>/dev/null || true
    { date '+%F %T %z'; printf '%s\\n' \"\$line\"; } > \"\$root/${marker_name}\"
    test -s \"\$root/train/best_model.pt\"
  "
}

evaluate_one() {
  local key=$1
  local run=${REMOTE_RUN[$key]}
  local checkpoint_dir="${QUEUE_ROOT}/checkpoints/${run}"
  local checkpoint="${checkpoint_dir}/best_model.pt"
  local stamp eval_dir rc

  mkdir -p "${checkpoint_dir}"
  log "${key}: copying best checkpoint from ${REMOTE}"
  if ! rsync -az --partial "${REMOTE}:${REMOTE_RUN_ROOT}/${run}/train/best_model.pt" "${checkpoint}"; then
    log "${key}: checkpoint copy failed; will retry"
    return 1
  fi
  if [[ ! -s "${checkpoint}" ]]; then
    log "${key}: copied checkpoint is empty; will retry"
    return 1
  fi

  while [[ -n "$(gpu_compute_pids)" ]]; do
    log "${key}: local GPU ${GPU} is busy; waiting"
    sleep 60
  done

  stamp=$(date '+%Y%m%d_%H%M%S')
  eval_dir="${ADAPTER_ROOT}/runs/eval_exact_${key}_target20_5video_${stamp}"
  mkdir -p "${eval_dir}"
  printf '%s\n' "${eval_dir}" > "${QUEUE_ROOT}/state/${key}.eval_dir"
  log "${key}: starting target20 evaluation in ${eval_dir}"

  RUN_DIR="${eval_dir}" \
  CHECKPOINT="${checkpoint}" \
  GPU="${GPU}" \
  PORT=2043 \
  TM_PORT=8043 \
  bash "${ADAPTER_ROOT}/configs/eval_exact_adapter_target20_local.sh" \
    > "${eval_dir}/eval.log" 2>&1
  rc=$?
  echo "${rc}" > "${eval_dir}/eval.exit_code"
  if [[ ${rc} -eq 0 ]]; then
    touch "${QUEUE_ROOT}/state/${key}.evaluated"
    log "${key}: evaluation completed successfully"
    return 0
  fi

  log "${key}: evaluation exited with rc=${rc}; preserving results and marking attempted"
  touch "${QUEUE_ROOT}/state/${key}.attempted"
  return 1
}

log "watcher started; remote=${REMOTE}, poll=${POLL_SEC}s, local_gpu=${GPU}"

while true; do
  pending=0
  ready_key=""
  ready_time=0

  for key in v4_canonical v9_nocanonical; do
    if [[ -f "${QUEUE_ROOT}/state/${key}.evaluated" || -f "${QUEUE_ROOT}/state/${key}.attempted" ]]; then
      continue
    fi
    pending=$((pending + 1))
    status=$(remote_status "${key}" 2>&1) || status="SSH_ERROR ${status}"
    case "${status}" in
      READY\ *)
        completed_at=${status#READY }
        if [[ -z "${ready_key}" || ${completed_at} -lt ${ready_time} ]]; then
          ready_key=${key}
          ready_time=${completed_at}
        fi
        ;;
      STOP_REQUESTED)
        if [[ "${key}" == "v4_canonical" ]]; then
          stop_epoch=011
          stop_marker=STOPPED_AFTER_EPOCH11_NO_IMPROVEMENT
        else
          stop_epoch=014
          stop_marker=STOPPED_AFTER_EPOCH14_NO_IMPROVEMENT
        fi
        if stop_after_no_improvement "${key}" "${stop_epoch}" "${stop_marker}"; then
          log "${key}: stopped after epoch ${stop_epoch}; checkpoint is ready for evaluation"
        else
          log "${key}: failed to stop cleanly after epoch ${stop_epoch}; will retry"
        fi
        ;;
      FAILED*)
        if [[ ! -f "${QUEUE_ROOT}/state/${key}.train_failed_warned" ]]; then
          log "${key}: training session ended without summary/checkpoint"
          touch "${QUEUE_ROOT}/state/${key}.train_failed_warned"
        fi
        ;;
      RUNNING)
        ;;
      *)
        log "${key}: status check returned: ${status}"
        ;;
    esac
  done

  if [[ ${pending} -eq 0 ]]; then
    log "all queued evaluations finished or were attempted; watcher exiting"
    exit 0
  fi
  if [[ -n "${ready_key}" ]]; then
    evaluate_one "${ready_key}" || true
    continue
  fi
  sleep "${POLL_SEC}"
done
