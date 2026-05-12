import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_json_gz(path: Path) -> Dict:
    if not path.exists():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _episode_dirs(root: Path) -> Iterable[Path]:
    if (root / "frames.jsonl").exists():
        yield root
        return
    yield from sorted(path for path in root.glob("episode_*") if (path / "frames.jsonl").exists())


def _replace_symlink(path: Path, target: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def _copy_optional_file(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _selected_frame(source_episode: Path, frame: Dict, profile: str, include_measurements: bool) -> Dict:
    profile_tokens = frame.get("profile_tokens", {})
    if profile not in profile_tokens:
        raise KeyError(f"Missing profile_tokens[{profile!r}] in {source_episode / 'frames.jsonl'}")
    tokens = profile_tokens[profile]
    selected = dict(frame)
    selected["sensor_profile"] = profile
    selected["primary_profile"] = profile
    selected["source_episode_dir"] = str(source_episode)
    selected["camera_tokens"] = dict(tokens.get("camera_tokens", {}))
    selected["rgb_token"] = tokens.get("rgb_token")
    selected["lidar_token"] = tokens.get("lidar_token")
    selected["lidar_laz_token"] = tokens.get("lidar_laz_token")
    selected["lidar_bev_token"] = tokens.get("lidar_bev_token")

    if include_measurements:
        measurement_token = selected.get("measurement_token")
        if measurement_token:
            measurement = _read_json_gz(source_episode / measurement_token)
            imu = measurement.get("imu")
            if imu and "imu" not in selected:
                selected["imu"] = imu
            for key in ("target_point", "target_point_next", "command", "command_onehot", "route_angle"):
                if key in measurement and key not in selected:
                    selected[key] = measurement[key]
    return selected


def export_profile_view(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    episodes = list(_episode_dirs(input_root))
    if not episodes:
        raise RuntimeError(f"No paired episodes found under {input_root}")

    exported = []
    skipped = []
    for source_episode in episodes:
        summary = _read_json(source_episode / "episode_summary.json")
        motion = summary.get("motion", {}) if isinstance(summary, dict) else {}
        if args.skip_invalid_motion and motion and motion.get("motion_valid") is False:
            skipped.append({"source": str(source_episode), "reason": "invalid_motion", "motion": motion})
            continue

        target_episode = output_root / source_episode.name
        if target_episode.exists() and args.overwrite:
            shutil.rmtree(target_episode)
        target_episode.mkdir(parents=True, exist_ok=True)

        rig_layout = source_episode / "rigs" / args.profile / "sensor_layout.json"
        if not rig_layout.exists():
            raise FileNotFoundError(f"Missing rig layout for profile {args.profile!r}: {rig_layout}")
        _copy_optional_file(rig_layout, target_episode / "sensor_layout.json")
        for name in ("episode_meta.json", "episode_summary.json", "sensor_layouts.json"):
            _copy_optional_file(source_episode / name, target_episode / name)

        if args.symlink_assets:
            for name in ("rigs", "measurements"):
                source = source_episode / name
                if source.exists():
                    _replace_symlink(target_episode / name, source)

        frames = _read_jsonl(source_episode / "frames.jsonl")
        kept = 0
        with (target_episode / "frames.jsonl").open("w", encoding="utf-8") as handle:
            for frame in frames:
                selected = _selected_frame(source_episode, frame, args.profile, args.include_measurements)
                if args.require_cameras:
                    missing = [camera for camera in args.require_cameras if camera not in selected.get("camera_tokens", {})]
                    if missing:
                        continue
                if not selected.get("lidar_bev_token"):
                    continue
                handle.write(json.dumps(selected, ensure_ascii=False) + "\n")
                kept += 1
        exported.append({"source": str(source_episode), "episode": str(target_episode), "frames": kept})

    meta = {
        "source_root": str(input_root),
        "output_root": str(output_root),
        "profile": args.profile,
        "episodes": exported,
        "skipped": skipped,
        "symlink_assets": bool(args.symlink_assets),
    }
    _write_json(output_root / "profile_view_meta.json", meta)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "profile": args.profile,
                "episodes": len(exported),
                "skipped": len(skipped),
                "frames": sum(item["frames"] for item in exported),
            },
            indent=2,
        ),
        flush=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a lightweight single-profile view from a paired TransFuser++ collection.")
    parser.add_argument("--input-root", required=True, help="Paired collection root containing episode_*/profile_tokens.")
    parser.add_argument("--output-root", required=True, help="Output root with rewritten episode_*/frames.jsonl.")
    parser.add_argument("--profile", required=True, help="Profile name under frame.profile_tokens and rigs/<profile>/sensor_layout.json.")
    parser.add_argument("--require-cameras", default="", help="Comma-separated camera names that must be present.")
    parser.add_argument("--include-measurements", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--symlink-assets", action=argparse.BooleanOptionalAction, default=True, help="Symlink rigs/ and measurements/ instead of copying heavy data.")
    parser.add_argument("--skip-invalid-motion", action="store_true", help="Skip episodes whose episode_summary.motion.motion_valid is false.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.require_cameras = [item.strip() for item in args.require_cameras.split(",") if item.strip()]
    export_profile_view(args)


if __name__ == "__main__":
    main()
