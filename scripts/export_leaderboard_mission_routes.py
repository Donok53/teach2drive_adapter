#!/usr/bin/env python3
"""Split long CARLA Leaderboard routes into short one-scenario mission routes."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _install_carla_path(carla_root: str) -> None:
    root = Path(carla_root).expanduser()
    api_root = root / "PythonAPI" / "carla"
    if api_root.exists() and str(api_root) not in sys.path:
        sys.path.insert(0, str(api_root))

    try:
        import carla  # noqa: F401

        return
    except Exception:
        pass

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    py_name = f"py{sys.version_info.major}.{sys.version_info.minor}"
    dist = root / "PythonAPI" / "carla" / "dist"
    candidates = [
        *[path for path in sorted(dist.glob("carla-*.egg")) if py_name in path.name or py_tag in path.name],
        *[path for path in sorted(dist.glob("carla-*.whl")) if py_tag in path.name],
    ]
    for path in reversed(candidates):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import carla  # noqa: F401


def _indent(element: ET.Element, level: int = 0) -> None:
    pad = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = pad + "  "
        for child in element:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not element.tail or not element.tail.strip()):
        element.tail = pad


def _point_from_attrs(attrs: dict[str, str]) -> tuple[float, float, float]:
    return (float(attrs["x"]), float(attrs["y"]), float(attrs.get("z", 0.0)))


def _distance_xy(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cumulative(points: list[tuple[float, float, float]]) -> list[float]:
    distances = [0.0]
    for prev, cur in zip(points, points[1:]):
        distances.append(distances[-1] + _distance_xy(prev, cur))
    return distances


def _point_at_s(
    points: list[tuple[float, float, float]],
    distances: list[float],
    target_s: float,
) -> tuple[float, float, float]:
    if target_s <= 0:
        return points[0]
    if target_s >= distances[-1]:
        return points[-1]
    for idx in range(1, len(points)):
        if distances[idx] >= target_s:
            span = max(distances[idx] - distances[idx - 1], 1e-6)
            ratio = (target_s - distances[idx - 1]) / span
            prev = points[idx - 1]
            cur = points[idx]
            return (
                prev[0] + (cur[0] - prev[0]) * ratio,
                prev[1] + (cur[1] - prev[1]) * ratio,
                prev[2] + (cur[2] - prev[2]) * ratio,
            )
    return points[-1]


def _nearest_route_position(
    points: list[tuple[float, float, float]],
    distances: list[float],
    target: tuple[float, float, float],
) -> tuple[float, float, float]:
    best_s = 0.0
    best_d = float("inf")
    for idx in range(1, len(points)):
        a = points[idx - 1]
        b = points[idx]
        vx = b[0] - a[0]
        vy = b[1] - a[1]
        wx = target[0] - a[0]
        wy = target[1] - a[1]
        denom = vx * vx + vy * vy
        ratio = 0.0 if denom <= 1e-9 else max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
        px = a[0] + vx * ratio
        py = a[1] + vy * ratio
        d = math.hypot(target[0] - px, target[1] - py)
        if d < best_d:
            best_d = d
            best_s = distances[idx - 1] + (distances[idx] - distances[idx - 1]) * ratio
    return best_s, best_d, _point_at_s(points, distances, best_s)


def _sample_slice(
    points: list[tuple[float, float, float]],
    start_s: float,
    end_s: float,
    spacing_m: float,
    max_waypoints: int,
) -> list[tuple[float, float, float]]:
    distances = _cumulative(points)
    selected = [_point_at_s(points, distances, start_s)]
    next_s = start_s + spacing_m
    while next_s < end_s - 1e-6:
        selected.append(_point_at_s(points, distances, next_s))
        next_s += spacing_m
    selected.append(_point_at_s(points, distances, end_s))

    deduped = [selected[0]]
    for point in selected[1:]:
        if _distance_xy(point, deduped[-1]) > 1.0:
            deduped.append(point)

    if max_waypoints > 1 and len(deduped) > max_waypoints:
        keep = [0]
        span = len(deduped) - 1
        for idx in range(1, max_waypoints - 1):
            keep.append(round(idx * span / (max_waypoints - 1)))
        keep.append(span)
        deduped = [deduped[idx] for idx in sorted(set(keep))]
    return deduped


def _load_world_map(args: argparse.Namespace, town: str):
    _install_carla_path(args.carla_root)
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    current = world.get_map().name
    if args.load_world and not current.endswith(f"/{town}") and f"/{town}/" not in current:
        available = client.get_available_maps()
        load_name = town
        for name in available:
            if name.endswith(f"/{town}") or name.endswith(f"/{town}/{town}"):
                load_name = name
                break
        world = client.load_world(load_name)
    return world.get_map(), carla


def _planner_points(
    route_positions: list[tuple[float, float, float]],
    args: argparse.Namespace,
    town: str,
) -> list[tuple[float, float, float]]:
    road_map, carla = _load_world_map(args, town)
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    planner = GlobalRoutePlanner(road_map, args.planner_resolution_m)
    dense: list[tuple[float, float, float]] = []
    for start, end in zip(route_positions, route_positions[1:]):
        start_loc = carla.Location(x=start[0], y=start[1], z=start[2] + 0.5)
        end_loc = carla.Location(x=end[0], y=end[1], z=end[2] + 0.5)
        segment = planner.trace_route(start_loc, end_loc)
        for waypoint, _road_option in segment:
            loc = waypoint.transform.location
            point = (float(loc.x), float(loc.y), float(loc.z))
            if not dense or _distance_xy(point, dense[-1]) > 0.5:
                dense.append(point)
    if len(dense) < 2:
        raise RuntimeError(f"Planner returned fewer than two points for {town}")
    return dense


def _format_point(point: tuple[float, float, float]) -> dict[str, str]:
    return {"x": f"{point[0]:.3f}", "y": f"{point[1]:.3f}", "z": f"{point[2]:.3f}"}


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value[:80] or "scenario"


def _write_route(
    output_path: Path,
    route_id: int,
    town: str,
    weathers: ET.Element | None,
    waypoints: list[tuple[float, float, float]],
    scenario: ET.Element,
) -> None:
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", {"id": str(route_id), "town": town})
    if weathers is not None:
        route.append(copy.deepcopy(weathers))
    waypoint_node = ET.SubElement(route, "waypoints")
    for point in waypoints:
        ET.SubElement(waypoint_node, "position", _format_point(point))
    scenarios_node = ET.SubElement(route, "scenarios")
    scenarios_node.append(copy.deepcopy(scenario))
    _indent(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


def export_missions(args: argparse.Namespace) -> list[dict]:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_re = re.compile(args.scenario_regex) if args.scenario_regex else None
    exclude_re = re.compile(args.exclude_scenario_regex) if args.exclude_scenario_regex else None

    manifest: list[dict] = []
    route_id = args.route_id_start
    for source_xml in args.routes_xml:
        source_path = Path(source_xml).expanduser()
        root = ET.parse(source_path).getroot()
        for route in root.findall("route"):
            source_route_id = route.get("id", "0")
            town = route.get("town", args.town or "Town13")
            route_positions = [_point_from_attrs(node.attrib) for node in route.find("waypoints").findall("position")]
            if len(route_positions) < 2:
                continue
            if args.route_mode == "planner":
                dense_points = _planner_points(route_positions, args, town)
            else:
                dense_points = route_positions
            distances = _cumulative(dense_points)
            scenarios_node = route.find("scenarios")
            if scenarios_node is None:
                continue
            for scenario_index, scenario in enumerate(scenarios_node.findall("scenario")):
                scenario_name = scenario.get("name", f"scenario_{scenario_index:03d}")
                scenario_type = scenario.get("type", "Unknown")
                label = f"{scenario_type}:{scenario_name}"
                if scenario_re and not scenario_re.search(label):
                    continue
                if exclude_re and exclude_re.search(label):
                    continue
                trigger_node = scenario.find("trigger_point")
                if trigger_node is None:
                    continue
                trigger = _point_from_attrs(trigger_node.attrib)
                trigger_s, nearest_d, _nearest = _nearest_route_position(dense_points, distances, trigger)
                start_s = max(0.0, trigger_s - args.pre_distance_m)
                end_s = min(distances[-1], trigger_s + args.post_distance_m)
                if end_s - start_s < args.min_route_distance_m:
                    pad = (args.min_route_distance_m - (end_s - start_s)) / 2.0
                    start_s = max(0.0, start_s - pad)
                    end_s = min(distances[-1], end_s + pad)
                mission_points = _sample_slice(dense_points, start_s, end_s, args.spacing_m, args.max_waypoints)
                if len(mission_points) < 2:
                    continue

                mission_name = f"mission_{len(manifest):03d}_src{source_route_id}_s{scenario_index:03d}_{_safe_name(scenario_type)}_{_safe_name(scenario_name)}"
                output_path = output_dir / f"{mission_name}.xml"
                _write_route(output_path, route_id, town, route.find("weathers"), mission_points, scenario)
                record = {
                    "mission_index": len(manifest),
                    "route_id": route_id,
                    "route_xml": str(output_path),
                    "source_xml": str(source_path),
                    "source_route_id": source_route_id,
                    "scenario_index": scenario_index,
                    "scenario_name": scenario_name,
                    "scenario_type": scenario_type,
                    "town": town,
                    "trigger": {"x": trigger[0], "y": trigger[1], "z": trigger[2], "yaw": trigger_node.attrib.get("yaw")},
                    "trigger_route_s_m": trigger_s,
                    "nearest_route_distance_m": nearest_d,
                    "route_start_s_m": start_s,
                    "route_end_s_m": end_s,
                    "route_length_m": end_s - start_s,
                    "waypoint_count": len(mission_points),
                }
                manifest.append(record)
                route_id += 1
                if args.limit > 0 and len(manifest) >= args.limit:
                    break
            if args.limit > 0 and len(manifest) >= args.limit:
                break
        if args.limit > 0 and len(manifest) >= args.limit:
            break

    manifest_path = output_dir / "manifest.jsonl"
    list_path = output_dir / "mission_routes.txt"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    list_path.write_text("\n".join(record["route_xml"] for record in manifest) + ("\n" if manifest else ""), encoding="utf-8")
    print(json.dumps({"missions": len(manifest), "manifest": str(manifest_path), "mission_list": str(list_path)}, indent=2))
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes-xml", nargs="+", required=True, help="Input CARLA Leaderboard route XML files")
    parser.add_argument("--output-dir", required=True, help="Directory for generated one-scenario XML files")
    parser.add_argument("--carla-root", default="/home/byeongjae/carla-simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--town", default=None)
    parser.add_argument("--route-mode", choices=("planner", "polyline"), default="planner")
    parser.add_argument("--no-load-world", action="store_false", dest="load_world")
    parser.set_defaults(load_world=True)
    parser.add_argument("--planner-resolution-m", type=float, default=2.0)
    parser.add_argument("--pre-distance-m", type=float, default=80.0)
    parser.add_argument("--post-distance-m", type=float, default=140.0)
    parser.add_argument("--min-route-distance-m", type=float, default=140.0)
    parser.add_argument("--spacing-m", type=float, default=25.0)
    parser.add_argument("--max-waypoints", type=int, default=18)
    parser.add_argument("--route-id-start", type=int, default=10000)
    parser.add_argument("--scenario-regex", default=None, help="Regex matched against 'type:name'")
    parser.add_argument("--exclude-scenario-regex", default=None, help="Regex matched against 'type:name'")
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> None:
    export_missions(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
