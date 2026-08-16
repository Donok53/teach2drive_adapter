#!/usr/bin/env bash
set -u

RUN_NAME=${RUN_NAME:-eval_pdm_oracle_tfpp_pid_smoke5_20260814}
HOST_DATA_ROOT=${HOST_DATA_ROOT:-/media/aimlab/HDD00/users/byeongjae/workspace/teach2drive/data}
ADAPTER_ROOT=${ADAPTER_ROOT:-/media/aimlab/HDD00/users/byeongjae/workspace/teach2drive/code/teach2drive_adapter}
TOTAL=${TOTAL:-20}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-10}
INITIAL_PID=${INITIAL_PID:-}
SUMMARY=${HOST_DATA_ROOT}/runs/${RUN_NAME}/summary.tsv

if [[ -n "${INITIAL_PID}" ]]; then
  while kill -0 "${INITIAL_PID}" 2>/dev/null; do
    sleep 15
  done
fi

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  next_index=$(python3 - "${SUMMARY}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
best = -1
if path.exists():
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                best = max(best, int(row["index"]))
            except (KeyError, TypeError, ValueError):
                pass
print(best + 1)
PY
  )
  if (( next_index >= TOTAL )); then
    echo "[$(date --iso-8601=seconds)] oracle evaluation complete rows=${next_index}/${TOTAL}"
    exit 0
  fi

  echo "[$(date --iso-8601=seconds)] resume attempt=${attempt} start=${next_index} remaining=$((TOTAL-next_index))"
  set +e
  (
    cd "${ADAPTER_ROOT}" || exit 1
    env RUN_NAME="${RUN_NAME}" GPU=0 PORT=2063 TM_PORT=8063 \
      START_INDEX="${next_index}" LIMIT="$((TOTAL-next_index))" \
      VIDEO_RECORD_INDICES=0,1,2,3,4 \
      bash scripts/run_pdm_oracle_tfpp_pid_dl2.sh
  )
  status=$?
  set -e
  echo "[$(date --iso-8601=seconds)] attempt=${attempt} exit=${status}"
  sleep 10
done

echo "[$(date --iso-8601=seconds)] supervisor exhausted MAX_ATTEMPTS=${MAX_ATTEMPTS}" >&2
exit 2
