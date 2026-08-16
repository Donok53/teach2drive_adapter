from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _summarize_trace(path: Path) -> dict[str, float | int]:
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    turn = [row for row in rows if float(row.get("turn_gate", 0.0)) > 0.2]
    active = [row for row in rows if float(row.get("gate", 0.0)) > 1e-6]
    final_delta = [abs(float(row.get("adapter_delta", 0.0))) for row in rows]
    return {
        "records": len(rows),
        "duration_sec": float(rows[-1].get("timestamp", 0.0)) if rows else 0.0,
        "active_fraction": len(active) / max(len(rows), 1),
        "turn_fraction": len(turn) / max(len(rows), 1),
        "mean_abs_applied_delta": _safe_mean(final_delta),
        "maximum_abs_applied_delta": max(final_delta, default=0.0),
        "turn_mean_abs_applied_delta": _safe_mean(
            [abs(float(row.get("adapter_delta", 0.0))) for row in turn]
        ),
        "turn_mean_desired_actual_yaw_error": _safe_mean(
            [abs(float(row["desired_yaw_rate"]) - float(row["yaw_rate"])) for row in turn]
        ),
        "turn_mean_calibration_gain": _safe_mean(
            [float(row.get("calibration_gain", 1.0)) for row in turn]
        ),
        "turn_risk_gate_off_fraction": _safe_mean(
            [float(row.get("risk_gate", 0.0) == 0.0) for row in turn]
        ),
        "turn_overshoot_gate_off_fraction": _safe_mean(
            [float(row.get("overshoot_gate", 0.0) == 0.0) for row in turn]
        ),
        "turn_exit_fraction": _safe_mean(
            [float(row.get("exit_strength", 0.0) >= 0.5) for row in turn]
        ),
    }


def _outcome(path: Path) -> dict[str, str | float | int]:
    if not path.exists():
        return {"outcome": "MISSING"}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        return {"outcome": "MISSING"}
    row = rows[-1]
    return {
        "outcome": row["outcome"],
        "route": float(row["score_route"]),
        "composed": float(row["score_composed"]),
        "infractions": int(row["num_infractions"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize closed-loop TF++ dynamics-v2 traces")
    parser.add_argument("--root", required=True)
    parser.add_argument("--missions", default="2,4")
    parser.add_argument("--blends", default="000,025,050")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    report = []
    for mission in (int(value) for value in args.missions.split(",") if value):
        for blend in (value.strip() for value in args.blends.split(",") if value.strip()):
            run = root / f"mission_{mission}_b{blend}"
            trace = run / "traces" / "dynamics_v2.jsonl"
            record = {"mission": mission, "blend": int(blend) / 100.0, **_outcome(run / "summary.tsv")}
            if trace.exists():
                record.update(_summarize_trace(trace))
            report.append(record)
    output = json.dumps(report, indent=2, sort_keys=True)
    print(output)
    if args.out:
        Path(args.out).expanduser().write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
