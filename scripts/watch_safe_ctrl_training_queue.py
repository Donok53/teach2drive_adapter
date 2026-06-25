#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import time
from dataclasses import dataclass


DEFAULT_LOG_DIR = pathlib.Path(
    os.environ.get(
        "SAFE_CTRL_LOG_DIR",
        "/home/jovyan/teach2drive/logs/benchmix8_safe_ctrl_conservative_queue_target_only",
    )
)


@dataclass
class Launch:
    gpu: int
    recipe: str
    session: str
    log_path: pathlib.Path


def run_stdout(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        return ""
    return proc.stdout


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def age(path: pathlib.Path) -> str:
    if not path.exists():
        return "-"
    seconds = max(0, int(time.time() - path.stat().st_mtime))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def trim(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def tmux_sessions() -> set[str]:
    sessions = set()
    for line in run_stdout(["tmux", "ls"]).splitlines():
        if ":" in line:
            sessions.add(line.split(":", 1)[0])
    return sessions


def gpu_stats() -> dict[int, str]:
    output = run_stdout(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    stats: dict[int, str] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            stats[int(parts[0])] = f"{int(parts[1])}MB/{int(parts[2])}%"
        except ValueError:
            continue
    return stats


def read_scheduler(log_dir: pathlib.Path) -> tuple[dict[str, str], list[Launch]]:
    scheduler = log_dir / "scheduler.log"
    meta: dict[str, str] = {}
    launches: list[Launch] = []
    if not scheduler.exists():
        return meta, launches
    launch_re = re.compile(r"LAUNCH gpu=(\d+) recipe=(\S+) session=(\S+) log=(.+)$")
    for line in scheduler.read_text(errors="replace").splitlines():
        if line.startswith("safe_ctrl queue start "):
            for key in ("epoch1_threshold", "epoch2_threshold", "baseline_safe_ctrl", "epochs", "recipe_set", "recipes"):
                match = re.search(rf"{key}=([^ ]+)", line)
                if match:
                    meta[key] = match.group(1)
            gpus = re.search(r"gpus=(\[[^\]]+\])", line)
            if gpus:
                meta["gpus"] = gpus.group(1)
        match = launch_re.search(line)
        if match:
            launches.append(
                Launch(
                    gpu=int(match.group(1)),
                    recipe=match.group(2),
                    session=match.group(3),
                    log_path=pathlib.Path(match.group(4)),
                )
            )
    return meta, launches


def read_decisions(log_dir: pathlib.Path) -> list[dict[str, object]]:
    path = log_dir / "safe_ctrl_queue_decisions.jsonl"
    if not path.exists():
        return []
    decisions: list[dict[str, object]] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            decisions.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return decisions


def parse_last_step(text: str) -> dict[str, str]:
    matches = re.findall(r"^step=(\d+)/(\d+) .*$", text, flags=re.MULTILINE)
    if not matches:
        return {}
    step, total = matches[-1]
    line_match = re.findall(r"^(step=" + re.escape(step) + r"/" + re.escape(total) + r" .*)$", text, flags=re.MULTILINE)
    line = line_match[-1] if line_match else ""
    result = {"step": f"{int(step)}/{int(total)}", "pct": f"{100.0 * int(step) / max(int(total), 1):.1f}%"}
    for key in ("loss", "ctrl", "samples/s"):
        match = re.search(rf"{re.escape(key)}=([0-9.]+)", line)
        if match:
            result[key] = match.group(1)
    hinge = re.search(r"hinge=([0-9.]+)/([0-9.]+)/([0-9.]+)", line)
    if hinge:
        result["hinge"] = "/".join(f"{float(v):.3f}" for v in hinge.groups())
    act = re.search(r"act=([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)", line)
    if act:
        result["act"] = "/".join(f"{float(v):.3f}" for v in act.groups())
    soft = re.search(r"soft=([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)", line)
    if soft:
        result["soft"] = "/".join(f"{float(v):.3f}" for v in soft.groups())
    return result


def parse_last_epoch(text: str) -> dict[str, str]:
    matches = re.findall(r"^(epoch=\d+ .*)$", text, flags=re.MULTILINE)
    if not matches:
        return {}
    line = matches[-1]
    result: dict[str, str] = {}
    epoch = re.search(r"epoch=(\d+)", line)
    if epoch:
        result["epoch"] = str(int(epoch.group(1)))
    for key in ("safe_ctrl", "closed", "ctrl_closed", "safe", "val"):
        match = re.search(rf"{key}=([0-9.]+)", line)
        if match:
            result[key] = match.group(1)
    select = re.search(r"select=safety_controller_closed_loop_proxy:([0-9.]+)", line)
    if select and "safe_ctrl" not in result:
        result["safe_ctrl"] = select.group(1)
    plan_sgtb = re.search(r"plan_sgtb=([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)", line)
    if plan_sgtb:
        result["plan_sgtb"] = "/".join(f"{float(v):.3f}" for v in plan_sgtb.groups())
    hinge = re.search(r"hinge=([0-9.]+)/([0-9.]+)/([0-9.]+)", line)
    if hinge:
        result["hinge"] = "/".join(f"{float(v):.3f}" for v in hinge.groups())
    act = re.search(r"act=([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)", line)
    if act:
        result["act"] = "/".join(f"{float(v):.3f}" for v in act.groups())
    soft = re.search(r"soft=([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)", line)
    if soft:
        result["soft"] = "/".join(f"{float(v):.3f}" for v in soft.groups())
    return result


def parse_train_log(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {}
    result = parse_last_step(text)
    epoch = parse_last_epoch(text)
    for key, value in epoch.items():
        result[f"epoch_{key}" if key in result else key] = value
    return result


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in rows:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def render(log_dir: pathlib.Path) -> None:
    meta, launches = read_scheduler(log_dir)
    decisions = read_decisions(log_dir)
    sessions = tmux_sessions()
    gpus = gpu_stats()
    scheduler = log_dir / "scheduler.log"

    print(f"[{utc_now()}] safe_ctrl training queue")
    print(f"log_dir={log_dir} scheduler_age={age(scheduler)}")
    if meta:
        print(
            "queue "
            + " ".join(
                f"{key}={meta[key]}"
                for key in ("threshold", "epochs", "gpus", "recipe_set", "recipes")
                if key in meta
            )
        )
    print()

    decided_sessions = {str(item.get("session", "")) for item in decisions}
    active_rows: list[list[str]] = []
    for launch in launches:
        if launch.session in decided_sessions:
            continue
        info = parse_train_log(launch.log_path)
        running = launch.session in sessions
        state = "RUN" if running else "ENDED?"
        active_rows.append(
            [
                str(launch.gpu),
                gpus.get(launch.gpu, "-"),
                state,
                trim(launch.recipe, 34),
                info.get("step", "-"),
                info.get("pct", "-"),
                info.get("loss", "-"),
                info.get("safe_ctrl", "-"),
                info.get("plan_sgtb", "-"),
                info.get("hinge", "-"),
                info.get("soft", "-"),
                info.get("act", "-"),
                info.get("samples/s", "-"),
                age(launch.log_path),
            ]
        )

    print("active")
    if active_rows:
        print_table(
            ["gpu", "gpu_use", "state", "recipe", "step", "%", "loss", "safe_ctrl", "plan_s/g/t/b", "hinge g/t/p", "soft s/g/h/f", "act g/s/t/b", "s/s", "age"],
            active_rows,
        )
    else:
        print("(no active training jobs)")
    print()

    print("decisions")
    if decisions:
        rows: list[list[str]] = []
        for item in decisions[-10:]:
            rows.append(
                [
                    str(item.get("time_utc", "-")).replace("T", " ").replace("Z", ""),
                    str(item.get("gpu", "-")),
                    trim(str(item.get("recipe", "-")), 34),
                    f"{float(item.get('epoch1_safe_ctrl', 0.0)):.6f}" if "epoch1_safe_ctrl" in item else "-",
                    str(item.get("decision", "-")),
                ]
            )
        print_table(["time", "gpu", "recipe", "epoch1_safe_ctrl", "decision"], rows)
    else:
        print("(no epoch-1 decisions yet)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch safe_ctrl recipe queue training progress.")
    parser.add_argument("--log-dir", type=pathlib.Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    while True:
        if args.watch and not args.no_clear:
            print("\033[2J\033[H", end="")
        render(args.log_dir.expanduser())
        if not args.watch:
            break
        time.sleep(max(float(args.interval), 1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
