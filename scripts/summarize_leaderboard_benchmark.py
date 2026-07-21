#!/usr/bin/env python3
"""Create benchmark-style tables and HTML reports from CARLA Leaderboard runs.

This is intentionally compatible with the local mission-level runs produced by
``scripts/run_tfpp_mission_batch_autorestart.sh`` while exposing metrics that
look closer to Bench2Drive/Leaderboard papers: driving score, route completion,
success rate, ability groups, and infraction breakdowns.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ABILITY = {
    "Overtaking": [
        "Accident",
        "AccidentTwoWays",
        "ConstructionObstacle",
        "ConstructionObstacleTwoWays",
        "HazardAtSideLaneTwoWays",
        "HazardAtSideLane",
        "ParkedObstacleTwoWays",
        "ParkedObstacle",
        "VehicleOpensDoorTwoWays",
    ],
    "Merging": [
        "CrossingBicycleFlow",
        "EnterActorFlow",
        "HighwayExit",
        "InterurbanActorFlow",
        "HighwayCutIn",
        "InterurbanAdvancedActorFlow",
        "MergerIntoSlowTrafficV2",
        "MergerIntoSlowTraffic",
        "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionRightTurn",
        "NonSignalizedJunctionLeftTurnEnterFlow",
        "ParkingExit",
        "SequentialLaneChange",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "SignalizedJunctionLeftTurnEnterFlow",
    ],
    "Emergency_Brake": [
        "BlockedIntersection",
        "DynamicObjectCrossing",
        "HardBreakRoute",
        "OppositeVehicleTakingPriority",
        "OppositeVehicleRunningRedLight",
        "ParkingCutIn",
        "PedestrianCrossing",
        "ParkingCrossingPedestrian",
        "StaticCutIn",
        "VehicleTurningRoute",
        "VehicleTurningRoutePedestrian",
        "ControlLoss",
    ],
    "Give_Way": ["InvadingTurn", "YieldToEmergencyVehicle"],
    "Traffic_Signs": [
        "BlockedIntersection",
        "OppositeVehicleTakingPriority",
        "OppositeVehicleRunningRedLight",
        "PedestrianCrossing",
        "VehicleTurningRoute",
        "VehicleTurningRoutePedestrian",
        "EnterActorFlow",
        "CrossingBicycleFlow",
        "NonSignalizedJunctionLeftTurn",
        "NonSignalizedJunctionRightTurn",
        "NonSignalizedJunctionLeftTurnEnterFlow",
        "SignalizedJunctionLeftTurn",
        "SignalizedJunctionRightTurn",
        "SignalizedJunctionLeftTurnEnterFlow",
        "T_Junction",
        "VanillaNonSignalizedTurn",
        "VanillaSignalizedTurnEncounterGreenLight",
        "VanillaSignalizedTurnEncounterRedLight",
        "VanillaNonSignalizedTurnEncounterStopsign",
    ],
}

NON_CRITICAL_INFRACTION_KEYS = {"min_speed_infractions"}
MISSION_RE = re.compile(r"mission_\d+_src[^_]+_s\d+_([^_]+)_")


@dataclass
class MissionRow:
    run_label: str
    index: int
    mission: str
    scenario_type: str
    outcome: str
    status: str
    score_route: float
    score_penalty: float
    score_composed: float
    num_infractions: int
    critical_infractions: int
    b2d_success: bool
    checkpoint: str
    video: str
    infractions: dict[str, int]


@dataclass
class RunMetrics:
    label: str
    run_dir: str
    total: int
    pass_count: int
    fail_count: int
    invalid_count: int
    pass_rate: float
    b2d_success_count: int
    b2d_success_rate: float
    driving_score: float
    route_completion: float
    infraction_penalty: float
    infractions_per_route: float
    critical_infractions_per_route: float
    warning: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _scenario_type_from_mission(mission: str) -> str:
    match = MISSION_RE.search(mission)
    if match:
        return match.group(1)
    parts = mission.split("_")
    return parts[0] if parts else "unknown"


def _read_summary_rows(run_dir: Path) -> list[dict[str, str]]:
    summary = run_dir / "summary.tsv"
    if not summary.exists():
        return []
    with summary.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _result_jsons(run_dir: Path) -> list[Path]:
    results = run_dir / "results"
    return sorted(results.glob("*.json")) if results.exists() else []


def _last_record(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    records = data.get("_checkpoint", {}).get("records", [])
    return records[-1] if records else {}


def _infraction_counts(record: dict[str, Any]) -> dict[str, int]:
    infractions = record.get("infractions", {}) or {}
    counts: dict[str, int] = {}
    for key, value in infractions.items():
        counts[key] = len(value) if isinstance(value, list) else int(bool(value))
    return counts


def _is_b2d_success(record: dict[str, Any]) -> bool:
    status = str(record.get("status", ""))
    if status not in {"Completed", "Perfect"}:
        return False
    for key, count in _infraction_counts(record).items():
        if key not in NON_CRITICAL_INFRACTION_KEYS and count > 0:
            return False
    return True


def _rows_from_run(run_dir: Path, label: str) -> list[MissionRow]:
    summary_rows = _read_summary_rows(run_dir)
    by_mission = {row.get("mission", ""): row for row in summary_rows}
    rows: list[MissionRow] = []

    jsons = _result_jsons(run_dir)
    if jsons:
        for result_path in jsons:
            mission = result_path.stem
            record = _last_record(result_path)
            scores = record.get("scores", {}) if record else {}
            summary = by_mission.get(mission, {})
            infractions = _infraction_counts(record)
            critical = sum(
                count
                for key, count in infractions.items()
                if key not in NON_CRITICAL_INFRACTION_KEYS
            )
            video = summary.get("video", "")
            if not video:
                candidates = sorted((run_dir / "videos").glob(f"{mission}_*"))
                video = str(candidates[0]) if candidates else ""
            rows.append(
                MissionRow(
                    run_label=label,
                    index=_safe_int(summary.get("index"), len(rows)),
                    mission=mission,
                    scenario_type=_scenario_type_from_mission(mission),
                    outcome=summary.get("outcome", ""),
                    status=str(record.get("status", summary.get("status", "UNKNOWN"))),
                    score_route=_safe_float(scores.get("score_route", summary.get("score_route"))),
                    score_penalty=_safe_float(scores.get("score_penalty", summary.get("score_penalty"))),
                    score_composed=_safe_float(scores.get("score_composed", summary.get("score_composed"))),
                    num_infractions=_safe_int(record.get("num_infractions", summary.get("num_infractions"))),
                    critical_infractions=critical,
                    b2d_success=_is_b2d_success(record),
                    checkpoint=str(result_path),
                    video=video,
                    infractions=infractions,
                )
            )
    else:
        for summary in summary_rows:
            mission = summary.get("mission", "")
            rows.append(
                MissionRow(
                    run_label=label,
                    index=_safe_int(summary.get("index"), len(rows)),
                    mission=mission,
                    scenario_type=_scenario_type_from_mission(mission),
                    outcome=summary.get("outcome", ""),
                    status=summary.get("status", "UNKNOWN"),
                    score_route=_safe_float(summary.get("score_route")),
                    score_penalty=_safe_float(summary.get("score_penalty")),
                    score_composed=_safe_float(summary.get("score_composed")),
                    num_infractions=_safe_int(summary.get("num_infractions")),
                    critical_infractions=_safe_int(summary.get("num_infractions")),
                    b2d_success=summary.get("outcome") == "PASS",
                    checkpoint=summary.get("checkpoint", ""),
                    video=summary.get("video", ""),
                    infractions={},
                )
            )

    return sorted(rows, key=lambda row: row.index)


def _run_metrics(run_dir: Path, label: str, rows: list[MissionRow]) -> RunMetrics:
    total = len(rows)
    outcomes = Counter(row.outcome for row in rows)
    b2d_success = sum(row.b2d_success for row in rows)
    warning = ""
    if total != 220:
        warning = (
            f"local run has {total} routes; official Bench2Drive metrics assume 220 routes"
        )
    return RunMetrics(
        label=label,
        run_dir=str(run_dir),
        total=total,
        pass_count=outcomes.get("PASS", 0),
        fail_count=outcomes.get("FAIL", 0),
        invalid_count=outcomes.get("INVALID", 0),
        pass_rate=100.0 * outcomes.get("PASS", 0) / total if total else 0.0,
        b2d_success_count=b2d_success,
        b2d_success_rate=100.0 * b2d_success / total if total else 0.0,
        driving_score=_mean([row.score_composed for row in rows]),
        route_completion=_mean([row.score_route for row in rows]),
        infraction_penalty=_mean([row.score_penalty for row in rows]),
        infractions_per_route=_mean([float(row.num_infractions) for row in rows]),
        critical_infractions_per_route=_mean([float(row.critical_infractions) for row in rows]),
        warning=warning,
    )


def _scenario_metrics(rows: list[MissionRow]) -> list[dict[str, Any]]:
    grouped: dict[str, list[MissionRow]] = defaultdict(list)
    for row in rows:
        grouped[row.scenario_type].append(row)
    result = []
    for scenario, items in sorted(grouped.items()):
        result.append(
            {
                "scenario_type": scenario,
                "total": len(items),
                "pass_rate": 100.0 * sum(item.outcome == "PASS" for item in items) / len(items),
                "b2d_success_rate": 100.0 * sum(item.b2d_success for item in items) / len(items),
                "driving_score": _mean([item.score_composed for item in items]),
                "route_completion": _mean([item.score_route for item in items]),
                "critical_infractions": sum(item.critical_infractions for item in items),
            }
        )
    return result


def _ability_metrics(rows: list[MissionRow]) -> list[dict[str, Any]]:
    result = []
    for ability, scenarios in ABILITY.items():
        items = [row for row in rows if row.scenario_type in scenarios]
        if not items:
            result.append(
                {
                    "ability": ability,
                    "total": 0,
                    "success_rate": 0.0,
                    "driving_score": 0.0,
                }
            )
            continue
        result.append(
            {
                "ability": ability,
                "total": len(items),
                "success_rate": 100.0 * sum(row.b2d_success for row in items) / len(items),
                "driving_score": _mean([row.score_composed for row in items]),
            }
        )
    return result


def _infraction_metrics(rows: list[MissionRow]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.infractions)
    return [
        {
            "infraction": key,
            "count": count,
            "per_route": count / len(rows) if rows else 0.0,
            "critical": key not in NON_CRITICAL_INFRACTION_KEYS,
        }
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _bar(value: float, max_value: float = 100.0) -> str:
    pct = 0.0 if max_value <= 0 else max(0.0, min(100.0, value / max_value * 100.0))
    return (
        '<div class="bar"><span style="width: '
        f"{pct:.1f}%"
        f'"></span></div><b>{_fmt(value)}</b>'
    )


def _rel_link(path: str, out_dir: Path, label: str) -> str:
    if not path:
        return ""
    target = Path(path)
    href = html.escape(str(Path("../../") / target.name)) if not target.is_absolute() else html.escape(str(target))
    try:
        href = html.escape(str(target.relative_to(out_dir)))
    except Exception:
        try:
            href = html.escape(str(Path(path).resolve().relative_to(out_dir.resolve())))
        except Exception:
            try:
                href = html.escape(str(Path(path).resolve().relative_to(out_dir.resolve().parent)))
            except Exception:
                href = html.escape(str(path))
    return f'<a href="{href}">{html.escape(label)}</a>'


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _write_html(
    path: Path,
    run_metrics: list[RunMetrics],
    rows_by_label: dict[str, list[MissionRow]],
    scenario_by_label: dict[str, list[dict[str, Any]]],
    ability_by_label: dict[str, list[dict[str, Any]]],
    infractions_by_label: dict[str, list[dict[str, Any]]],
) -> None:
    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 28px; color: #20242a; }
    h1 { margin-bottom: 4px; }
    h2 { margin-top: 28px; }
    .note { color: #5f6670; margin: 0 0 18px; }
    table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }
    th, td { border-bottom: 1px solid #e1e4e8; padding: 8px 10px; text-align: left; vertical-align: top; }
    th { background: #f5f7fa; font-weight: 700; }
    tr.fail td { background: #fff7f4; }
    tr.invalid td { background: #fff4f4; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
    .card { border: 1px solid #dde2e8; border-radius: 8px; padding: 12px; background: #fbfcfd; }
    .card .label { color: #66707a; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
    .card .value { font-size: 24px; font-weight: 750; margin-top: 4px; }
    .bar { display: inline-block; width: 110px; height: 8px; background: #e8edf3; border-radius: 999px; overflow: hidden; margin-right: 8px; }
    .bar span { display: block; height: 100%; background: #276ef1; }
    .warn { color: #a35b00; font-weight: 650; }
    .small { font-size: 12px; color: #66707a; }
    """

    overall_rows = []
    for metric in run_metrics:
        overall_rows.append(
            [
                html.escape(metric.label),
                str(metric.total),
                _bar(metric.driving_score),
                _bar(metric.route_completion),
                _bar(metric.b2d_success_rate),
                _bar(metric.pass_rate),
                _fmt(metric.infraction_penalty),
                _fmt(metric.critical_infractions_per_route),
                html.escape(metric.warning),
            ]
        )

    cards = ""
    if run_metrics:
        best = max(run_metrics, key=lambda item: item.driving_score)
        cards = f"""
        <div class="grid">
          <div class="card"><div class="label">Best Run</div><div class="value">{html.escape(best.label)}</div></div>
          <div class="card"><div class="label">Driving Score</div><div class="value">{_fmt(best.driving_score)}</div></div>
          <div class="card"><div class="label">Route Completion</div><div class="value">{_fmt(best.route_completion)}</div></div>
          <div class="card"><div class="label">Bench2Drive-style Success</div><div class="value">{_fmt(best.b2d_success_rate)}%</div></div>
        </div>
        """

    sections = []
    for metric in run_metrics:
        label = metric.label
        scenario_rows = [
            [
                html.escape(row["scenario_type"]),
                str(row["total"]),
                _bar(float(row["driving_score"])),
                _bar(float(row["b2d_success_rate"])),
                _fmt(float(row["route_completion"])),
                str(row["critical_infractions"]),
            ]
            for row in scenario_by_label[label]
        ]
        ability_rows = [
            [
                html.escape(row["ability"]),
                str(row["total"]),
                _bar(float(row["success_rate"])),
                _bar(float(row["driving_score"])),
            ]
            for row in ability_by_label[label]
        ]
        infraction_rows = [
            [
                html.escape(row["infraction"]),
                str(row["count"]),
                _fmt(float(row["per_route"])),
                "yes" if row["critical"] else "no",
            ]
            for row in infractions_by_label[label]
        ]
        mission_rows = []
        for row in rows_by_label[label]:
            cls = row.outcome.lower()
            mission_rows.append(
                [
                    str(row.index),
                    html.escape(row.mission),
                    html.escape(row.scenario_type),
                    html.escape(row.outcome),
                    html.escape(row.status),
                    _fmt(row.score_route),
                    _fmt(row.score_penalty),
                    _fmt(row.score_composed),
                    str(row.critical_infractions),
                    _rel_link(row.video, path.parent, "video") if row.video else "",
                    _rel_link(row.checkpoint, path.parent, "json") if row.checkpoint else "",
                    cls,
                ]
            )
        mission_table = "<table><thead><tr>" + "".join(
            f"<th>{html.escape(h)}</th>"
            for h in [
                "#",
                "Mission",
                "Scenario",
                "Outcome",
                "Status",
                "Route",
                "Penalty",
                "Score",
                "Critical Infractions",
                "Video",
                "JSON",
            ]
        ) + "</tr></thead><tbody>"
        for row in mission_rows:
            cls = row.pop()
            mission_table += (
                f'<tr class="{html.escape(cls)}">'
                + "".join(f"<td>{cell}</td>" for cell in row)
                + "</tr>"
            )
        mission_table += "</tbody></table>"

        sections.append(
            f"""
            <h2>{html.escape(label)}</h2>
            <p class="small">{html.escape(metric.run_dir)}</p>
            <h3>Ability Groups</h3>
            {_html_table(["Ability", "N", "Success", "Driving Score"], ability_rows)}
            <h3>Scenario Breakdown</h3>
            {_html_table(["Scenario", "N", "Driving Score", "Success", "Route", "Critical Infractions"], scenario_rows)}
            <h3>Infraction Breakdown</h3>
            {_html_table(["Infraction", "Count", "Per Route", "Critical"], infraction_rows)}
            <h3>Mission Details</h3>
            {mission_table}
            """
        )

    body = f"""<!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Leaderboard Benchmark Report</title>
      <style>{css}</style>
    </head>
    <body>
      <h1>Leaderboard Benchmark Report</h1>
      <p class="note">Local CARLA Leaderboard-style closed-loop evaluation. If N is not 220, Bench2Drive-style rates are normalized over this local run, not the official Bench2Drive denominator.</p>
      {cards}
      <h2>Overall Comparison</h2>
      {_html_table(["Run", "N", "Driving Score", "Route Completion", "B2D-style Success", "Strict Pass", "Penalty", "Critical Infractions / Route", "Warning"], overall_rows)}
      {''.join(sections)}
    </body>
    </html>
    """
    path.write_text(body, encoding="utf-8")


def _parse_run_specs(specs: list[str]) -> list[tuple[Path, str]]:
    parsed = []
    for spec in specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
            parsed.append((Path(path).expanduser(), label))
        else:
            path = Path(spec).expanduser()
            parsed.append((path, path.name))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run directory, or label=/path/to/run. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Defaults to <single-run>/benchmark_report or ./benchmark_report.",
    )
    args = parser.parse_args()

    run_specs = _parse_run_specs(args.run)
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser()
    elif len(run_specs) == 1:
        out_dir = run_specs[0][0] / "benchmark_report"
    else:
        out_dir = Path("benchmark_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[RunMetrics] = []
    rows_by_label: dict[str, list[MissionRow]] = {}
    scenario_by_label: dict[str, list[dict[str, Any]]] = {}
    ability_by_label: dict[str, list[dict[str, Any]]] = {}
    infractions_by_label: dict[str, list[dict[str, Any]]] = {}

    for run_dir, label in run_specs:
        rows = _rows_from_run(run_dir, label)
        metrics = _run_metrics(run_dir, label, rows)
        all_metrics.append(metrics)
        rows_by_label[label] = rows
        scenario_by_label[label] = _scenario_metrics(rows)
        ability_by_label[label] = _ability_metrics(rows)
        infractions_by_label[label] = _infraction_metrics(rows)

        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
        _write_csv(out_dir / f"{stem}_missions.csv", [asdict(row) for row in rows])
        _write_csv(out_dir / f"{stem}_scenarios.csv", scenario_by_label[label])
        _write_csv(out_dir / f"{stem}_abilities.csv", ability_by_label[label])
        _write_csv(out_dir / f"{stem}_infractions.csv", infractions_by_label[label])

    _write_csv(out_dir / "overall.csv", [asdict(metric) for metric in all_metrics])
    summary = {
        "overall": [asdict(metric) for metric in all_metrics],
        "scenarios": scenario_by_label,
        "abilities": ability_by_label,
        "infractions": infractions_by_label,
    }
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_html(
        out_dir / "benchmark_report.html",
        all_metrics,
        rows_by_label,
        scenario_by_label,
        ability_by_label,
        infractions_by_label,
    )
    print(f"wrote {out_dir / 'benchmark_report.html'}")


if __name__ == "__main__":
    main()
