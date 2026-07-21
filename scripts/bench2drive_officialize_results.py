#!/usr/bin/env python3
"""Prepare Bench2Drive route results for official metric aggregation.

Bench2Drive's official metric scripts expect route JSON files containing one
checkpoint record per evaluated route. If CARLA/evaluator crashes before a
record is written, this helper writes a zero-score Failed record so the route is
counted in the official 220-route denominator instead of disappearing.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


INFRACTION_KEYS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "min_speed_infractions",
    "yield_emergency_vehicle_infractions",
    "scenario_timeouts",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
)


def route_paths(route_list: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in route_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summary_indices(summary_tsv: Path) -> set[int]:
    if not summary_tsv.exists():
        return set()
    with summary_tsv.open("r", encoding="utf-8") as handle:
        return {
            int(row["index"])
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("index", "").strip().isdigit()
        }


def route_info(route_xml: Path) -> tuple[str, str, str]:
    root = ET.parse(route_xml).getroot()
    route = root.find("route")
    if route is None:
        raise ValueError(f"{route_xml} has no <route>")
    route_id = route.get("id") or route_xml.stem
    town = route.get("town") or ""
    scenario = ""
    scenarios = route.find("scenarios")
    if scenarios is not None:
        scenario_elem = scenarios.find("scenario")
        if scenario_elem is not None:
            scenario = scenario_elem.get("type") or ""
    return route_id, town, scenario


def has_records(checkpoint: Path) -> bool:
    if not checkpoint.exists():
        return False
    try:
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
    except Exception:
        return False
    records = data.get("_checkpoint", {}).get("records", [])
    return bool(records)


def placeholder_record(index: int, route_xml: Path) -> dict:
    route_id, town, scenario = route_info(route_xml)
    return {
        "index": index,
        "route_id": f"RouteScenario_{route_id}_rep0",
        "status": "Failed",
        "num_infractions": 0,
        "infractions": {key: [] for key in INFRACTION_KEYS},
        "scores": {
            "score_route": 0,
            "score_penalty": 0,
            "score_composed": 0,
        },
        "meta": {
            "route_xml": str(route_xml),
            "route_id": route_id,
            "town": town,
            "scenario": scenario,
            "placeholder_reason": "missing_or_empty_checkpoint_record",
        },
    }


def write_placeholder(checkpoint: Path, index: int, route_xml: Path) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "_checkpoint": {
            "global_record": {},
            "progress": [0, 1],
            "records": [placeholder_record(index, route_xml)],
        },
        "entry_status": "Finished",
        "eligible": True,
        "sensors": [],
        "values": [],
        "labels": [],
    }
    checkpoint.write_text(json.dumps(data, indent=2), encoding="utf-8")


def official_success(record: dict) -> bool:
    if record.get("status") not in {"Completed", "Perfect"}:
        return False
    infractions = record.get("infractions", {})
    for key, value in infractions.items():
        if key != "min_speed_infractions" and value:
            return False
    return True


def current_records(result_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(result_dir.glob("*.json")):
        if path.name == "merged.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records.extend(data.get("_checkpoint", {}).get("records", []))
    return records


def write_local_official_summary(result_dir: Path) -> Path:
    records = current_records(result_dir)
    driving_score = sum(float(r.get("scores", {}).get("score_composed", 0.0)) for r in records) / 220.0
    success_rate = sum(1 for r in records if official_success(r)) / 220.0
    out = {
        "eval_num": len(records),
        "driving score": driving_score,
        "success rate": success_rate,
        "note": "Bench2Drive official denominator is 220. Use merged.json after eval_num reaches 220.",
    }
    # Keep this out of result_dir/*.json because Bench2Drive's official
    # merge_route_json.py blindly reads every JSON file in that folder.
    out_path = result_dir.parent / "official_progress.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-list", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--summary-tsv", type=Path)
    parser.add_argument("--all-routes", action="store_true", help="fill placeholders for all 220 routes")
    parser.add_argument("--merge", action="store_true", help="run Bench2Drive's official merge_route_json.py")
    parser.add_argument(
        "--bench2drive-root",
        type=Path,
        default=Path("/home/byeongjae/code/carla_garage/Bench2Drive"),
    )
    args = parser.parse_args()

    routes = route_paths(args.route_list)
    if args.all_routes:
        wanted = set(range(len(routes)))
    elif args.summary_tsv:
        wanted = summary_indices(args.summary_tsv)
    else:
        wanted = {
            int(path.stem.rsplit("_", 1)[-1])
            for path in args.result_dir.glob("bench2drive_*.json")
            if path.stem.rsplit("_", 1)[-1].isdigit()
        }

    fixed = 0
    for idx in sorted(wanted):
        if idx < 0 or idx >= len(routes):
            continue
        checkpoint = args.result_dir / f"bench2drive_{idx:02d}.json"
        if has_records(checkpoint):
            continue
        write_placeholder(checkpoint, idx, routes[idx])
        fixed += 1

    stale_progress = args.result_dir / "official_progress.json"
    if stale_progress.exists():
        stale_progress.unlink()

    progress_path = write_local_official_summary(args.result_dir)

    if args.merge:
        merge_script = args.bench2drive_root / "tools" / "merge_route_json.py"
        subprocess.run([sys.executable, str(merge_script), "-f", str(args.result_dir)], check=True)

    print(f"result_dir={args.result_dir}")
    print(f"routes_considered={len(wanted)} placeholders_written={fixed}")
    print(f"official_progress={progress_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
