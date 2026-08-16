#!/usr/bin/env bash
set -u

MODE=${MODE:?Set MODE=checkpoint or speed}
GPU=${GPU:?Set GPU}
PORT=${PORT:?Set PORT}
TM_PORT=${TM_PORT:?Set TM_PORT}
RUN_NAME=${RUN_NAME:?Set RUN_NAME}
HOST_DATA_ROOT=${HOST_DATA_ROOT:-/media/aimlab/HDD00/users/byeongjae/workspace/teach2drive/data}
ADAPTER_ROOT=${ADAPTER_ROOT:-/media/aimlab/HDD00/users/byeongjae/workspace/teach2drive/code/teach2drive_adapter}
TOTAL=${TOTAL:-20}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-12}
INITIAL_PID=${INITIAL_PID:-}
SUMMARY=${HOST_DATA_ROOT}/runs/${RUN_NAME}/summary.tsv

if [[ -n "${INITIAL_PID}" ]]; then
  while kill -0 "${INITIAL_PID}" 2>/dev/null; do sleep 15; done
fi

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  next_index=$(python3 - "${SUMMARY}" <<'PY'
import csv, sys
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
    echo "[$(date --iso-8601=seconds)] ${MODE} ablation complete rows=${next_index}/${TOTAL}"
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] mode=${MODE} resume=${attempt} start=${next_index} remaining=$((TOTAL-next_index))"
  set +e
  (
    cd "${ADAPTER_ROOT}" || exit 1
    env MODE="${MODE}" GPU="${GPU}" PORT="${PORT}" TM_PORT="${TM_PORT}" RUN_NAME="${RUN_NAME}" \
      START_INDEX="${next_index}" LIMIT="$((TOTAL-next_index))" VIDEO_RECORD_INDICES=0,1,2,3,4 \
      bash scripts/run_mixed_pdm_oracle_ablation_dl2.sh
  )
  status=$?
  set -e
  echo "[$(date --iso-8601=seconds)] mode=${MODE} attempt=${attempt} exit=${status}"
  sleep 10
done
exit 2
