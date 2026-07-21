import argparse
import json
import math
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
    candidates = [
        *[
            path
            for path in sorted((root / "PythonAPI" / "carla" / "dist").glob("carla-*.egg"))
            if py_name in path.name or py_tag in path.name
        ],
        *[
            path
            for path in sorted((root / "PythonAPI" / "carla" / "dist").glob("carla-*.whl"))
            if py_tag in path.name
        ],
    ]
    for path in reversed(candidates):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import carla  # noqa: F401


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda item: (int(item.get("step", 0)), float(item.get("time", 0.0))))
    return records


def _route_town(meta: dict, fallback: str) -> str:
    raw = str(meta.get("map") or fallback)
    town = raw.rsplit("/", 1)[-1]
    if town.endswith("_Opt"):
        town = town[: -len("_Opt")]
    return town


def _cumulative_distance(points: list[tuple[float, float, float]]) -> list[float]:
    distances = [0.0]
    for prev, cur in zip(points, points[1:]):
        distances.append(distances[-1] + math.hypot(cur[0] - prev[0], cur[1] - prev[1]))
    return distances


def _select_by_spacing(points: list[tuple[float, float, float]], spacing_m: float, max_points: int) -> list[tuple[float, float, float]]:
    if len(points) <= 2:
        return points
    distances = _cumulative_distance(points)
    selected = [points[0]]
    next_distance = spacing_m
    for point, distance in zip(points[1:-1], distances[1:-1]):
        if distance + 1e-6 >= next_distance:
            if math.hypot(point[0] - selected[-1][0], point[1] - selected[-1][1]) > 1e-3:
                selected.append(point)
            next_distance += spacing_m
    if math.hypot(points[-1][0] - selected[-1][0], points[-1][1] - selected[-1][1]) > 1e-3:
        selected.append(points[-1])

    if max_points > 0 and len(selected) > max_points:
        keep = [0]
        span = len(selected) - 1
        for i in range(1, max_points - 1):
            keep.append(round(i * span / (max_points - 1)))
        keep.append(span)
        selected = [selected[i] for i in sorted(set(keep))]
    return selected


def _select_points(
    frames: list[dict],
    spacing_m: float,
    include_phases: set[str],
    max_points: int,
    max_distance_m: float,
) -> tuple[list[tuple[float, float, float]], dict]:
    filtered = []
    for frame in frames:
        if include_phases and str(frame.get("phase", "")) not in include_phases:
            continue
        odom = frame["odom"]
        filtered.append((float(odom["x"]), float(odom["y"]), float(odom.get("z", 0.0))))
    if len(filtered) < 2:
        raise ValueError("Need at least two route points after phase filtering")

    distances = _cumulative_distance(filtered)
    if max_distance_m > 0:
        cutoff = 0
        for idx, distance in enumerate(distances):
            cutoff = idx
            if distance >= max_distance_m:
                break
        filtered = filtered[: max(cutoff + 1, 2)]
        distances = _cumulative_distance(filtered)

    selected = _select_by_spacing(filtered, spacing_m, max_points)

    return selected, {
        "source_frames": len(frames),
        "filtered_frames": len(filtered),
        "source_distance_m": distances[-1],
        "selected_points": len(selected),
    }


def _carla_world(args: argparse.Namespace, town: str):
    _install_carla_path(args.carla_root)
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.carla_timeout)
    if args.carla_load_world:
        client.load_world(town)
    return client.get_world(), carla


def _snap_points_to_map(
    points: list[tuple[float, float, float]],
    args: argparse.Namespace,
    town: str,
) -> tuple[list[tuple[float, float, float]], dict]:
    world, carla = _carla_world(args, town)
    road_map = world.get_map()
    snapped = []
    offsets = []
    failures = 0
    for x, y, z in points:
        waypoint = road_map.get_waypoint(
            carla.Location(x=x, y=y, z=z + 0.5),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            failures += 1
            snapped.append((x, y, z))
            continue
        loc = waypoint.transform.location
        offsets.append(math.hypot(float(loc.x) - x, float(loc.y) - y))
        snapped.append((float(loc.x), float(loc.y), float(loc.z)))
    summary = {
        "snap_failures": failures,
        "snap_mean_offset_m": sum(offsets) / len(offsets) if offsets else 0.0,
        "snap_max_offset_m": max(offsets) if offsets else 0.0,
        "map": road_map.name,
    }
    return snapped, summary


def _planner_route(
    points: list[tuple[float, float, float]],
    args: argparse.Namespace,
    town: str,
) -> tuple[list[tuple[float, float, float]], dict]:
    world, carla = _carla_world(args, town)
    road_map = world.get_map()
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    planner = GlobalRoutePlanner(road_map, args.planner_resolution_m)
    start_raw = points[0]
    end_raw = points[-1]
    start_wp = road_map.get_waypoint(
        carla.Location(x=start_raw[0], y=start_raw[1], z=start_raw[2] + 0.5),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    end_wp = road_map.get_waypoint(
        carla.Location(x=end_raw[0], y=end_raw[1], z=end_raw[2] + 0.5),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if start_wp is None or end_wp is None:
        raise RuntimeError("Could not snap route start/end to driving lane")

    route = planner.trace_route(start_wp.transform.location, end_wp.transform.location)
    if len(route) < 2:
        raise RuntimeError("GlobalRoutePlanner returned fewer than two points")
    dense = [
        (
            float(waypoint.transform.location.x),
            float(waypoint.transform.location.y),
            float(waypoint.transform.location.z),
        )
        for waypoint, _road_option in route
    ]
    selected = _select_by_spacing(dense, args.spacing_m, args.max_points)
    start_offset = math.hypot(start_wp.transform.location.x - start_raw[0], start_wp.transform.location.y - start_raw[1])
    end_offset = math.hypot(end_wp.transform.location.x - end_raw[0], end_wp.transform.location.y - end_raw[1])
    summary = {
        "map": road_map.name,
        "planner_dense_points": len(dense),
        "planner_dense_distance_m": _cumulative_distance(dense)[-1],
        "planner_selected_points": len(selected),
        "planner_resolution_m": args.planner_resolution_m,
        "start_snap_offset_m": start_offset,
        "end_snap_offset_m": end_offset,
    }
    return selected, summary


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


def export_route(args: argparse.Namespace) -> dict:
    episode_dir = Path(args.episode_dir).expanduser()
    frames_path = episode_dir / "frames.jsonl"
    meta_path = episode_dir / "episode_meta.json"
    if not frames_path.exists():
        raise FileNotFoundError(frames_path)
    frames = _read_jsonl(frames_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    phases = {item.strip() for item in args.include_phases.split(",") if item.strip()}
    points, summary = _select_points(frames, args.spacing_m, phases, args.max_points, args.max_distance_m)
    town = args.town or _route_town(meta, "Town10HD")
    raw_start = points[0]
    raw_end = points[-1]

    if args.route_mode == "snap":
        points, route_summary = _snap_points_to_map(points, args, town)
        summary.update(route_summary)
    elif args.route_mode == "planner":
        points, route_summary = _planner_route(points, args, town)
        summary.update(route_summary)

    root = ET.Element("routes")
    route = ET.SubElement(root, "route", {"id": str(args.route_id), "town": town})
    weathers = ET.SubElement(route, "weathers")
    weather_base = {
        "cloudiness": str(args.cloudiness),
        "precipitation": str(args.precipitation),
        "precipitation_deposits": str(args.precipitation_deposits),
        "wetness": str(args.wetness),
        "wind_intensity": str(args.wind_intensity),
        "sun_azimuth_angle": str(args.sun_azimuth_angle),
        "fog_density": str(args.fog_density),
    }
    ET.SubElement(weathers, "weather", {**weather_base, "route_percentage": "0", "sun_altitude_angle": str(args.start_sun_altitude_angle)})
    ET.SubElement(
        weathers,
        "weather",
        {**weather_base, "route_percentage": "100", "sun_altitude_angle": str(args.end_sun_altitude_angle)},
    )
    waypoints = ET.SubElement(route, "waypoints")
    for x, y, z in points:
        ET.SubElement(
            waypoints,
            "position",
            {
                "x": f"{x:.6f}",
                "y": f"{y:.6f}",
                "z": f"{z:.6f}" if (args.keep_z or args.route_mode in {"snap", "planner"}) else "0.0",
            },
        )
    ET.SubElement(route, "scenarios")

    tree = ET.ElementTree(root)
    _indent(root)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="UTF-8", xml_declaration=True)

    summary.update(
        {
            "episode_dir": str(episode_dir),
            "output": str(output),
            "route_id": str(args.route_id),
            "town": route.attrib["town"],
            "include_phases": sorted(phases),
            "spacing_m": args.spacing_m,
            "max_distance_m": args.max_distance_m,
            "keep_z": args.keep_z,
            "route_mode": args.route_mode,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "start": points[0],
            "end": points[-1],
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Teach2Drive episode trajectory as a CARLA Leaderboard route XML.")
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--route-id", default="1000")
    parser.add_argument("--town", default="")
    parser.add_argument("--include-phases", default="drive")
    parser.add_argument("--spacing-m", type=float, default=20.0)
    parser.add_argument("--max-distance-m", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--keep-z", action="store_true")
    parser.add_argument("--route-mode", choices=["recorded", "snap", "planner"], default="recorded")
    parser.add_argument("--carla-root", default="/home/byeongjae/carla-simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--carla-timeout", type=float, default=30.0)
    parser.add_argument("--carla-load-world", action="store_true")
    parser.add_argument("--planner-resolution-m", type=float, default=2.0)
    parser.add_argument("--cloudiness", type=float, default=5.0)
    parser.add_argument("--precipitation", type=float, default=0.0)
    parser.add_argument("--precipitation-deposits", type=float, default=0.0)
    parser.add_argument("--wetness", type=float, default=0.0)
    parser.add_argument("--wind-intensity", type=float, default=10.0)
    parser.add_argument("--sun-azimuth-angle", type=float, default=-1.0)
    parser.add_argument("--start-sun-altitude-angle", type=float, default=90.0)
    parser.add_argument("--end-sun-altitude-angle", type=float, default=45.0)
    parser.add_argument("--fog-density", type=float, default=2.0)
    args = parser.parse_args()
    print(json.dumps(export_route(args), indent=2))


if __name__ == "__main__":
    main()
