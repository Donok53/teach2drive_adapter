import argparse
from pathlib import Path

from .sensor_layout import canonical_transfuserpp_layout, perturb_layout, save_sensor_layout


def find_episode_dirs(root: Path):
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("episode_"))
    if candidates:
        return candidates
    if (root / "frames.jsonl").exists():
        return [root]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Write sensor_layout.json sidecars for Teach2Drive episodes.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-name", default="sensor_layout.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--camera-z-m", type=float, default=0.0)
    parser.add_argument("--lidar-yaw-deg", type=float, default=0.0)
    parser.add_argument("--lidar-z-m", type=float, default=0.0)
    args = parser.parse_args()

    root = Path(args.input_root).expanduser()
    episode_dirs = find_episode_dirs(root)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {root}")

    layout = canonical_transfuserpp_layout()
    layout = perturb_layout(
        layout,
        camera_yaw_deg=args.camera_yaw_deg,
        camera_pitch_deg=args.camera_pitch_deg,
        camera_z_m=args.camera_z_m,
        lidar_yaw_deg=args.lidar_yaw_deg,
        lidar_z_m=args.lidar_z_m,
    )

    written = 0
    skipped = 0
    for episode_dir in episode_dirs:
        output = episode_dir / args.output_name
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        if not args.dry_run:
            save_sensor_layout(layout, output)
        written += 1

    print(
        {
            "input_root": str(root),
            "episodes": len(episode_dirs),
            "output_name": args.output_name,
            "written": written,
            "skipped": skipped,
            "dry_run": bool(args.dry_run),
            "camera_yaw_deg": args.camera_yaw_deg,
            "camera_pitch_deg": args.camera_pitch_deg,
            "camera_z_m": args.camera_z_m,
            "lidar_yaw_deg": args.lidar_yaw_deg,
            "lidar_z_m": args.lidar_z_m,
        }
    )


if __name__ == "__main__":
    main()
