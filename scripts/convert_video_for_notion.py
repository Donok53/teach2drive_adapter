#!/usr/bin/env python3
"""Convert CARLA evaluation videos to a Notion/browser-friendly MP4."""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _imageio_ffmpeg_binaries() -> list[Path]:
    candidates: list[Path] = []
    try:
        import imageio_ffmpeg  # type: ignore

        candidates.append(Path(imageio_ffmpeg.get_ffmpeg_exe()))
    except Exception:
        pass

    home = Path.home()
    patterns = [
        home / ".local/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*",
        home / ".pyenv/versions/*/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*",
        home / "miniconda3/envs/*/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*",
        home / "anaconda3/envs/*/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*",
    ]
    for pattern in patterns:
        candidates.extend(Path(path) for path in glob.glob(str(pattern)))
    return candidates


def find_ffmpeg() -> Path:
    env_path = os.environ.get("FFMPEG") or os.environ.get("FFMPEG_BIN")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return path

    path_from_shell = shutil.which("ffmpeg")
    if path_from_shell:
        return Path(path_from_shell)

    for path in _imageio_ffmpeg_binaries():
        if path.exists() and os.access(path, os.X_OK) and path.is_file():
            return path

    raise FileNotFoundError(
        "ffmpeg not found. Install one of:\n"
        "  python -m pip install --user imageio-ffmpeg\n"
        "  sudo apt-get install ffmpeg\n"
        "or set FFMPEG=/path/to/ffmpeg"
    )


def video_info(ffmpeg: Path, path: Path) -> str:
    proc = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.stdout


def is_notion_compatible(ffmpeg: Path, path: Path) -> bool:
    info = video_info(ffmpeg, path)
    has_h264 = "Video: h264" in info
    has_yuv420p = "yuv420p" in info
    has_mp4_container = "Input #0, mov,mp4" in info or path.suffix.lower() == ".mp4"
    return has_mp4_container and has_h264 and has_yuv420p


def output_path_for(input_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    if input_path.suffix.lower() == ".mp4":
        return input_path
    return input_path.with_suffix(".mp4")


def convert(args: argparse.Namespace) -> Path:
    ffmpeg = find_ffmpeg()
    input_path = Path(args.input).expanduser()
    if not input_path.exists() or input_path.stat().st_size <= 0:
        raise FileNotFoundError(f"video is missing or empty: {input_path}")

    output_path = output_path_for(input_path, Path(args.output).expanduser() if args.output else None)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_compatible and output_path.exists() and output_path.stat().st_size > 0:
        try:
            if is_notion_compatible(ffmpeg, output_path):
                print(f"notion video already compatible: {output_path}")
                return output_path
        except Exception:
            pass

    temp_path = output_path.with_name(output_path.name + ".notion_tmp.mp4")
    if temp_path.exists():
        temp_path.unlink()

    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(temp_path),
    ]
    print(f"converting for notion: {input_path} -> {output_path}")
    subprocess.run(cmd, check=True)

    if temp_path.stat().st_size <= 0:
        raise RuntimeError(f"converted video is empty: {temp_path}")

    if output_path == input_path:
        temp_path.replace(output_path)
    else:
        if output_path.exists():
            output_path.unlink()
        temp_path.replace(output_path)

    if args.verify:
        if not is_notion_compatible(ffmpeg, output_path):
            raise RuntimeError(f"converted video is not H.264/yuv420p compatible: {output_path}")

    print(f"notion video saved: {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--crf", type=int, default=int(os.environ.get("VIDEO_NOTION_CRF", "23")))
    parser.add_argument("--preset", default=os.environ.get("VIDEO_NOTION_PRESET", "medium"))
    parser.add_argument("--skip-compatible", action="store_true", default=True)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.set_defaults(verify=True)
    try:
        convert(parser.parse_args())
    except Exception as exc:
        print(f"convert_video_for_notion.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
