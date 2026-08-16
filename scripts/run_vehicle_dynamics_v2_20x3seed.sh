#!/usr/bin/env bash
set -euo pipefail

ADAPTER_ROOT=${ADAPTER_ROOT:-/home/byeongjae/code/teach2drive_adapter}
DYNAMICS_CHECKPOINT=${DYNAMICS_CHECKPOINT:?Set DYNAMICS_CHECKPOINT}
CALIBRATOR_CHECKPOINT=${CALIBRATOR_CHECKPOINT:?Set CALIBRATOR_CHECKPOINT}
BLEND=${BLEND:?Set the selected dynamics blend}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ADAPTER_ROOT}/runs/eval_vehicle_dynamics_v2_20x3seed}
SEEDS=${SEEDS:-0,1,2}
GPU=${GPU:-1}
PORT=${PORT:-2043}
TM_PORT=${TM_PORT:-8043}

mkdir -p "${OUTPUT_ROOT}"

first_missing_mission() {
  local summary=$1
  python - "${summary}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
valid = set()
if path.exists():
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("outcome") in {"PASS", "FAIL"}:
                valid.add(int(row["index"]))
for index in range(20):
    if index not in valid:
        print(index)
        break
else:
    print(20)
PY
}

IFS=',' read -r -a seed_list <<< "${SEEDS}"
for seed in "${seed_list[@]}"; do
  run_dir="${OUTPUT_ROOT}/seed_${seed}"
  start=$(first_missing_mission "${run_dir}/summary.tsv")
  if (( start >= 20 )); then
    echo "SKIP seed=${seed}: 20 valid missions already exist"
    continue
  fi
  echo "START seed=${seed} mission=${start} blend=${BLEND} $(date '+%F %T')"
  DYNAMICS_CHECKPOINT="${DYNAMICS_CHECKPOINT}" \
  CALIBRATOR_CHECKPOINT="${CALIBRATOR_CHECKPOINT}" \
  TFPP_DYNAMICS_BLEND="${BLEND}" SEED="${seed}" RUN_DIR="${run_dir}" \
  START_INDEX="${start}" LIMIT="$((20 - start))" RECORD_VIDEO=0 VIDEO_RECORD_INDICES=999 \
  GPU="${GPU}" PORT="${PORT}" TM_PORT="${TM_PORT}" \
  bash "${ADAPTER_ROOT}/configs/eval_exact_vehicle_dynamics_v2_target20_local.sh"
  echo "END seed=${seed} $(date '+%F %T')"
done

python - "${OUTPUT_ROOT}" "${BLEND}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = {"blend": float(sys.argv[2]), "seeds": []}
for summary in sorted(root.glob("seed_*/summary.tsv")):
    latest = {}
    with summary.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["outcome"] in {"PASS", "FAIL"}:
                latest[int(row["index"])] = row
    rows = [latest[index] for index in sorted(latest)]
    report["seeds"].append({
        "seed": int(summary.parent.name.split("_")[-1]),
        "valid": len(rows),
        "pass": sum(row["outcome"] == "PASS" for row in rows),
        "mean_route": sum(float(row["score_route"]) for row in rows) / max(len(rows), 1),
        "mean_composed": sum(float(row["score_composed"]) for row in rows) / max(len(rows), 1),
        "infractions": sum(int(row["num_infractions"]) for row in rows),
    })
(root / "three_seed_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY
