"""Export a Teach2Drive sensor-profile dataset as a minimal CARLA Garage dataset.

CARLA Garage's TransFuser++ trainer expects route folders with rgb/lidar/
measurements subdirectories plus a results.json.gz file.  Teach2Drive paired
collections store the same driving labels in frames.jsonl/profile_tokens, but
with profile-specific rig folders.  This exporter creates a lightweight view
that can be passed directly to carla_garage/team_code/train.py when auxiliary
semantic/depth/box losses are disabled.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - keeps --help usable before deps are installed.
    np = None  # type: ignore[assignment]


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_json_gz(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _episode_dirs(root: Path) -> Iterable[Path]:
    if (root / "frames.jsonl").exists():
        yield root
        return
    episodes = sorted(path for path in root.glob("episode_*") if (path / "frames.jsonl").exists())
    if episodes:
        yield from episodes
        return
    yield from sorted(path for path in root.iterdir() if path.is_dir() and (path / "frames.jsonl").exists())


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "episode"


def _selected_frame(frame: Mapping, profile: str) -> Dict:
    if "profile_tokens" not in frame:
        return dict(frame)
    tokens = frame.get("profile_tokens", {}).get(profile)
    if not tokens:
        raise KeyError(f"frame does not contain profile_tokens[{profile!r}]")
    selected = dict(frame)
    selected["sensor_profile"] = profile
    selected["camera_tokens"] = dict(tokens.get("camera_tokens", {}))
    selected["rgb_token"] = tokens.get("rgb_token")
    selected["lidar_token"] = tokens.get("lidar_token")
    selected["lidar_laz_token"] = tokens.get("lidar_laz_token")
    selected["lidar_bev_token"] = tokens.get("lidar_bev_token")
    return selected


def _resolve_token(episode_dir: Path, token: Optional[str]) -> Optional[Path]:
    if not token:
        return None
    path = Path(token)
    if path.is_absolute():
        return path
    return episode_dir / path


def _replace_link_or_copy(source: Path, target: Path, copy: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if copy:
        shutil.copy2(source, target)
        return
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def _load_npz_points(path: Path) -> np.ndarray:
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required to convert Teach2Drive .npz LiDAR to CARLA Garage .laz. "
            "Use the training environment, install numpy, or collect with LIDAR_FORMAT=both."
        )
    arrays = np.load(path)
    if "points" in arrays.files:
        points = arrays["points"]
    elif "raw" in arrays.files:
        points = arrays["raw"]
    else:
        first = arrays.files[0]
        points = arrays[first]
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"invalid LiDAR array shape in {path}: {points.shape}")
    return points[:, :3]


def _write_lidar_laz_from_npz(source_npz: Path, target_laz: Path) -> None:
    try:
        import laspy
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "laspy is required to convert Teach2Drive .npz LiDAR to CARLA Garage .laz. "
            "Install laspy/lazrs, or collect with LIDAR_FORMAT=both."
        ) from exc

    points = _load_npz_points(source_npz)
    target_laz.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3)
    header.offsets = np.min(points, axis=0) if len(points) else np.zeros(3, dtype=np.float32)
    header.scales = np.asarray([0.01, 0.01, 0.01], dtype=np.float64)
    with laspy.open(target_laz, mode="w", header=header) as writer:
        record = laspy.ScaleAwarePointRecord.zeros(points.shape[0], header=header)
        record.x = points[:, 0]
        record.y = points[:, 1]
        record.z = points[:, 2]
        writer.write_points(record)


def _copy_episode(args: argparse.Namespace, episode_dir: Path, scenario_dir: Path, route_index: int) -> Dict:
    frames = list(_read_jsonl(episode_dir / "frames.jsonl"))
    if not frames:
        return {"episode": str(episode_dir), "frames": 0, "skipped": "empty"}

    summary = _read_json(episode_dir / "episode_summary.json")
    motion = summary.get("motion", {}) if isinstance(summary, dict) else {}
    if args.skip_invalid_motion and motion and motion.get("motion_valid") is False:
        return {"episode": str(episode_dir), "frames": 0, "skipped": "invalid_motion"}

    town = args.town
    route_name = f"{town}_{_safe_name(episode_dir.name)}_Rep0"
    if args.prefix_index:
        route_name = f"{town}_Route{route_index:04d}_{_safe_name(episode_dir.name)}_Rep0"
    route_dir = scenario_dir / route_name
    if route_dir.exists():
        if args.overwrite:
            shutil.rmtree(route_dir)
        else:
            return {"episode": str(episode_dir), "route": str(route_dir), "frames": -1, "skipped": "exists"}

    (route_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (route_dir / "lidar").mkdir(parents=True, exist_ok=True)
    (route_dir / "measurements").mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    for frame in frames:
        try:
            selected = _selected_frame(frame, args.profile)
        except KeyError:
            skipped += 1
            continue

        camera_tokens = selected.get("camera_tokens", {}) or {}
        image_path = _resolve_token(episode_dir, camera_tokens.get(args.camera) or selected.get("rgb_token"))
        measurement_path = _resolve_token(episode_dir, selected.get("measurement_token"))
        lidar_laz_path = _resolve_token(episode_dir, selected.get("lidar_laz_token"))
        lidar_npz_path = _resolve_token(episode_dir, selected.get("lidar_token"))
        if image_path is None or measurement_path is None:
            skipped += 1
            continue
        if not image_path.exists() or not measurement_path.exists():
            skipped += 1
            continue

        frame_name = f"{kept:04d}"
        _replace_link_or_copy(image_path, route_dir / "rgb" / f"{frame_name}.jpg", copy=args.copy_assets)
        _replace_link_or_copy(measurement_path, route_dir / "measurements" / f"{frame_name}.json.gz", copy=args.copy_assets)

        target_laz = route_dir / "lidar" / f"{frame_name}.laz"
        if lidar_laz_path is not None and lidar_laz_path.exists():
            _replace_link_or_copy(lidar_laz_path, target_laz, copy=args.copy_assets)
        elif args.convert_npz_lidar and lidar_npz_path is not None and lidar_npz_path.exists():
            _write_lidar_laz_from_npz(lidar_npz_path, target_laz)
        else:
            skipped += 1
            (route_dir / "rgb" / f"{frame_name}.jpg").unlink(missing_ok=True)
            (route_dir / "measurements" / f"{frame_name}.json.gz").unlink(missing_ok=True)
            continue
        kept += 1

    if kept <= 0:
        shutil.rmtree(route_dir, ignore_errors=True)
        return {"episode": str(episode_dir), "frames": 0, "skipped": "no_complete_frames"}

    results = {
        "status": "Completed",
        "scores": {"score_route": 100.0, "score_penalty": 1.0, "score_composed": 100.0},
        "num_infractions": 0,
        "infractions": {"min_speed_infractions": []},
    }
    _write_json_gz(route_dir / "results.json.gz", results)
    return {
        "episode": str(episode_dir),
        "route": str(route_dir),
        "frames": kept,
        "skipped_frames": skipped,
    }


def export_dataset(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    scenario_dir = output_root / args.scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    episodes = list(_episode_dirs(input_root))
    if not episodes:
        raise RuntimeError(f"No episode directories with frames.jsonl found under {input_root}")

    exported = []
    for route_index, episode_dir in enumerate(episodes):
        exported.append(_copy_episode(args, episode_dir, scenario_dir, route_index))

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "scenario_dir": str(scenario_dir),
        "profile": args.profile,
        "camera": args.camera,
        "episodes": exported,
    }
    (scenario_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "scenario_dir": str(scenario_dir),
                "routes": sum(1 for item in exported if item.get("frames", 0) > 0),
                "frames": sum(max(0, int(item.get("frames", 0))) for item in exported),
                "skipped": [item for item in exported if item.get("skipped")],
            },
            indent=2,
        ),
        flush=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Teach2Drive paired collection or exported profile-view root.")
    parser.add_argument("--output-root", required=True, help="Output root; a scenario subdirectory is created below it.")
    parser.add_argument("--scenario-name", default="teach2drive_profile", help="Scenario folder name under output-root.")
    parser.add_argument("--profile", default="front_triplet_shifted", help="Profile to select from profile_tokens when present.")
    parser.add_argument("--camera", default="front", help="Camera name to export as CARLA Garage rgb/front input.")
    parser.add_argument("--town", default="Town13", help="Town prefix for route folder names, e.g. Town13.")
    parser.add_argument("--prefix-index", action="store_true", help="Prefix route folders with a stable Route#### index.")
    parser.add_argument("--skip-invalid-motion", action="store_true")
    parser.add_argument("--copy-assets", action="store_true", help="Copy image/laz/measurement files instead of symlinking where possible.")
    parser.add_argument("--convert-npz-lidar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.town.startswith("Town"):
        args.town = f"Town{int(args.town):02d}"
    if not os.environ.get("PYTHONHASHSEED"):
        os.environ["PYTHONHASHSEED"] = "0"
    export_dataset(args)


if __name__ == "__main__":
    main()
