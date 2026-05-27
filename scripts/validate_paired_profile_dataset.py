#!/usr/bin/env python3
"""Validate that a Teach2Drive collection is usable for paired feature/fusion training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_CAMERA_REQUIREMENTS = {
    "tfpp_ego": ("front",),
    "front_triplet_shifted": ("left", "front", "right"),
}


def _episode_dirs(root: Path) -> list[Path]:
    if (root / "frames.jsonl").exists():
        return [root]
    return sorted(path for path in root.glob("episode_*") if (path / "frames.jsonl").exists())


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_frame(episode_dir: Path, line_no: int, frame: dict[str, Any], profiles: list[str]) -> None:
    profile_tokens = frame.get("profile_tokens")
    _require(isinstance(profile_tokens, dict), f"{episode_dir}/frames.jsonl:{line_no}: missing profile_tokens")

    for profile in profiles:
        tokens = profile_tokens.get(profile)
        _require(isinstance(tokens, dict), f"{episode_dir}/frames.jsonl:{line_no}: missing profile_tokens[{profile!r}]")

        camera_tokens = tokens.get("camera_tokens")
        _require(isinstance(camera_tokens, dict), f"{episode_dir}/frames.jsonl:{line_no}: {profile}: missing camera_tokens")
        for camera in DEFAULT_CAMERA_REQUIREMENTS.get(profile, ()):
            _require(
                bool(camera_tokens.get(camera)),
                f"{episode_dir}/frames.jsonl:{line_no}: {profile}: missing camera {camera!r}",
            )

        _require(
            bool(tokens.get("lidar_bev_token")),
            f"{episode_dir}/frames.jsonl:{line_no}: {profile}: missing lidar_bev_token",
        )


def validate_dataset(root: Path, profiles: list[str], max_frames: int) -> dict[str, Any]:
    root = root.expanduser()
    _require(root.exists(), f"dataset root does not exist: {root}")
    _require(root.is_dir(), f"dataset root is not a directory: {root}")

    episodes = _episode_dirs(root)
    _require(bool(episodes), f"no episode_*/frames.jsonl files found under {root}")

    frames_total = 0
    checked_total = 0
    per_episode = []

    for episode_dir in episodes:
        for profile in profiles:
            layout = episode_dir / "rigs" / profile / "sensor_layout.json"
            _require(layout.exists(), f"missing sensor layout: {layout}")

        frames_in_episode = 0
        checked_in_episode = 0
        for line_no, frame in _read_jsonl(episode_dir / "frames.jsonl"):
            frames_in_episode += 1
            if max_frames <= 0 or checked_total < max_frames:
                _validate_frame(episode_dir, line_no, frame, profiles)
                checked_in_episode += 1
                checked_total += 1

        _require(frames_in_episode > 0, f"empty frames.jsonl: {episode_dir / 'frames.jsonl'}")
        per_episode.append(
            {
                "episode": episode_dir.name,
                "frames": frames_in_episode,
                "checked_frames": checked_in_episode,
            }
        )
        frames_total += frames_in_episode

    return {
        "root": str(root.resolve()),
        "profiles": profiles,
        "episodes": len(episodes),
        "frames": frames_total,
        "checked_frames": checked_total,
        "valid": True,
        "episode_preview": per_episode[:5],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Dataset root containing paired episode_*/frames.jsonl files.")
    parser.add_argument("--profiles", default="tfpp_ego,front_triplet_shifted", help="Comma-separated required profiles.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Validate only the first N frames across the dataset. Default 0 validates all frames.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    if not profiles:
        raise SystemExit("--profiles must contain at least one profile")

    try:
        summary = validate_dataset(Path(args.root), profiles, args.max_frames)
    except Exception as exc:  # noqa: BLE001 - CLI should print compact validation failures.
        print(f"dataset validation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
