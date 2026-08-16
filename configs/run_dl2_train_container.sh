#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=${WORK_ROOT:-/media/aimlab/HDD00/users/byeongjae/workspace/teach2drive}
IMAGE=${IMAGE:-teach2drive-adapter:dl2}
GPUS=${GPUS:-all}

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

if [[ $# -eq 0 ]]; then
  set -- /bin/bash
fi

exec docker run --rm "${TTY_ARGS[@]}" \
  --gpus "${GPUS}" \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${WORK_ROOT}/code:/data/users/byeongjae/code" \
  -v "${WORK_ROOT}/data:/data/dataset/byeongjae" \
  -w /data/users/byeongjae/code/teach2drive_adapter \
  "${IMAGE}" "$@"
