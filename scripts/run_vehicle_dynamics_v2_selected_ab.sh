#!/usr/bin/env bash
set -euo pipefail

ADAPTER_ROOT=${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}
DYNAMICS_CHECKPOINT=${DYNAMICS_CHECKPOINT:?Set DYNAMICS_CHECKPOINT}
CALIBRATOR_CHECKPOINT=${CALIBRATOR_CHECKPOINT:?Set CALIBRATOR_CHECKPOINT}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ADAPTER_ROOT}/runs/eval_vehicle_dynamics_v2_selected_ab_20260814}
GPU=${GPU:-1}
PORT=${PORT:-2043}
TM_PORT=${TM_PORT:-8043}
MISSIONS=${MISSIONS:-2,4,5,8,11,15,17}
BLENDS=${BLENDS:-0,0.25,0.5}
VIDEO_MISSIONS=${VIDEO_MISSIONS:-2,4}

mkdir -p "${OUTPUT_ROOT}"

has_valid_result() {
  local summary=$1
  [[ -s "${summary}" ]] || return 1
  awk -F '\t' 'NR > 1 && ($3 == "PASS" || $3 == "FAIL") { found=1 } END { exit !found }' "${summary}"
}

contains_csv_value() {
  local csv=$1 value=$2
  [[ ",${csv}," == *",${value},"* ]]
}

IFS=',' read -r -a mission_list <<< "${MISSIONS}"
IFS=',' read -r -a blend_list <<< "${BLENDS}"

for mission in "${mission_list[@]}"; do
  for blend in "${blend_list[@]}"; do
    label=$(python - "${blend}" <<'PY'
import sys
print(f"b{int(round(float(sys.argv[1]) * 100)):03d}")
PY
)
    run_dir="${OUTPUT_ROOT}/mission_${mission}_${label}"
    if has_valid_result "${run_dir}/summary.tsv"; then
      echo "SKIP mission=${mission} blend=${blend}: valid result exists"
      continue
    fi
    video_indices=999
    if contains_csv_value "${VIDEO_MISSIONS}" "${mission}"; then
      video_indices=${mission}
    fi
    echo "START mission=${mission} blend=${blend} video=${video_indices} $(date '+%F %T')"
    set +e
    DYNAMICS_CHECKPOINT="${DYNAMICS_CHECKPOINT}" \
    CALIBRATOR_CHECKPOINT="${CALIBRATOR_CHECKPOINT}" \
    START_INDEX="${mission}" LIMIT=1 VIDEO_RECORD_INDICES="${video_indices}" \
    TFPP_DYNAMICS_BLEND="${blend}" RUN_DIR="${run_dir}" GPU="${GPU}" PORT="${PORT}" TM_PORT="${TM_PORT}" \
    CARLA_EXTRA_ARGS="-graphicsadapter=${GPU}" \
    bash "${ADAPTER_ROOT}/configs/eval_exact_vehicle_dynamics_v2_target20_local.sh"
    status=$?
    set -e
    echo "END mission=${mission} blend=${blend} status=${status} $(date '+%F %T')"
  done
done

python - "${OUTPUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for summary in sorted(root.glob("mission_*/summary.tsv")):
    with summary.open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    if not records:
        continue
    record = records[-1]
    parts = summary.parent.name.split("_")
    rows.append({
        "mission": int(parts[1]),
        "blend": int(parts[2][1:]) / 100.0,
        "outcome": record["outcome"],
        "route": float(record["score_route"]),
        "penalty": float(record["score_penalty"]),
        "composed": float(record["score_composed"]),
        "infractions": int(record["num_infractions"]),
        "summary": str(summary),
    })
rows.sort(key=lambda row: (row["mission"], row["blend"]))
(root / "selected_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
PY
