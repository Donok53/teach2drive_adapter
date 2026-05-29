#!/usr/bin/env python3
"""Build a balanced PDM-Lite/CARLA Garage route subset XML."""

import argparse
import copy
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_TOWNS = "Town01,Town02,Town03,Town04,Town05,Town10HD,Town12,Town13"


def _parse_towns(towns_csv):
    return [town.strip() for town in towns_csv.split(",") if town.strip()]


def _route_length_m(route_elem):
    positions = []
    waypoints = route_elem.find("waypoints")
    if waypoints is None:
        return 0.0
    for pos in waypoints.iter("position"):
        positions.append((
            float(pos.attrib.get("x", 0.0)),
            float(pos.attrib.get("y", 0.0)),
            float(pos.attrib.get("z", 0.0)),
        ))
    total = 0.0
    for a, b in zip(positions, positions[1:]):
        total += math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    return total


def _load_routes(route_root, towns):
    route_root = Path(route_root)
    pools = {town: [] for town in towns}
    skipped = []
    for xml_path in sorted(route_root.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as exc:
            skipped.append({"path": str(xml_path), "reason": str(exc)})
            continue
        for route in tree.getroot().iter("route"):
            town = route.attrib.get("town")
            if town not in pools:
                continue
            pools[town].append({
                "path": xml_path,
                "route": copy.deepcopy(route),
                "source_id": route.attrib.get("id", ""),
                "length_m": _route_length_m(route),
            })
    return pools, skipped


def _balanced_sample(pools, total_routes, seed):
    rng = random.Random(seed)
    towns = list(pools)
    for town in towns:
        rng.shuffle(pools[town])

    selected = []
    offsets = {town: 0 for town in towns}
    exhausted = set()
    while len(selected) < total_routes and len(exhausted) < len(towns):
        for town in towns:
            if len(selected) >= total_routes:
                break
            if town in exhausted:
                continue
            offset = offsets[town]
            if offset >= len(pools[town]):
                exhausted.add(town)
                continue
            selected.append((town, pools[town][offset]))
            offsets[town] += 1
    return selected


def build_subset(args):
    towns = _parse_towns(args.towns_csv)
    if not towns:
        raise SystemExit("No towns selected")

    total_routes = args.total_routes
    if total_routes <= 0:
        total_routes = int(math.ceil(args.target_hours * 3600.0 / args.avg_route_sec))

    pools, skipped = _load_routes(args.route_root, towns)
    selected = _balanced_sample(pools, total_routes, args.seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta_output = Path(args.meta_output) if args.meta_output else output.with_suffix(".json")

    root = ET.Element("routes")
    selected_meta = []
    for route_index, (town, item) in enumerate(selected):
        route = copy.deepcopy(item["route"])
        route.set("id", str(route_index))
        route.set("source_id", item["source_id"])
        route.set("source_file", str(Path(item["path"]).relative_to(Path(args.route_root))))
        root.append(route)
        selected_meta.append({
            "index": route_index,
            "town": town,
            "source_file": str(item["path"]),
            "source_id": item["source_id"],
            "length_m": item["length_m"],
        })

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="   ")
    except AttributeError:
        pass
    tree.write(output, encoding="utf-8", xml_declaration=True)

    counts = {town: 0 for town in towns}
    length_by_town = {town: 0.0 for town in towns}
    for town, item in selected:
        counts[town] += 1
        length_by_town[town] += item["length_m"]

    meta = {
        "output": str(output),
        "route_root": str(Path(args.route_root)),
        "towns": towns,
        "seed": args.seed,
        "target_hours": args.target_hours,
        "avg_route_sec": args.avg_route_sec,
        "requested_total_routes": total_routes,
        "selected_total_routes": len(selected),
        "counts_by_town": counts,
        "length_m_by_town": length_by_town,
        "total_length_m": sum(length_by_town.values()),
        "available_by_town": {town: len(pools[town]) for town in towns},
        "skipped": skipped,
        "selected": selected_meta,
    }
    meta_output.parent.mkdir(parents=True, exist_ok=True)
    meta_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "selected"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-output", default="")
    parser.add_argument("--towns-csv", default=DEFAULT_TOWNS)
    parser.add_argument("--target-hours", type=float, default=3.0)
    parser.add_argument("--avg-route-sec", type=float, default=28.0)
    parser.add_argument("--total-routes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    build_subset(parser.parse_args())


if __name__ == "__main__":
    main()
