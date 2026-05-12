#!/usr/bin/env python3
"""Record a CARLA hero-vehicle camera view to an mp4 file."""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _install_carla_path(carla_root: str | None) -> None:
    if not carla_root:
        return
    root = Path(carla_root).expanduser()
    api_root = root / "PythonAPI" / "carla"
    if api_root.exists() and str(api_root) not in sys.path:
        sys.path.insert(0, str(api_root))


def _find_hero(world):
    vehicles = world.get_actors().filter("vehicle.*")
    for vehicle in vehicles:
        if vehicle.attributes.get("role_name") == "hero":
            return vehicle
    return vehicles[0] if vehicles else None


def _view_transform(carla, view: str):
    if view == "topdown":
        return carla.Transform(carla.Location(x=0.0, y=0.0, z=35.0), carla.Rotation(pitch=-90.0))
    if view == "front":
        return carla.Transform(carla.Location(x=1.6, y=0.0, z=2.2), carla.Rotation(pitch=-5.0))
    return carla.Transform(carla.Location(x=-8.0, y=0.0, z=3.2), carla.Rotation(pitch=-15.0))


def _make_camera(world, carla, vehicle, args):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(args.width))
    blueprint.set_attribute("image_size_y", str(args.height))
    blueprint.set_attribute("fov", str(args.fov))
    if args.sensor_tick > 0:
        blueprint.set_attribute("sensor_tick", str(args.sensor_tick))
    return world.spawn_actor(blueprint, _view_transform(carla, args.view), attach_to=vehicle)


def _image_to_bgr(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3].copy()


def record(args: argparse.Namespace) -> None:
    _install_carla_path(args.carla_root)
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stopped = False

    def _stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"Waiting for hero vehicle on {args.host}:{args.port}...")
    vehicle = None
    start = time.monotonic()
    while not stopped:
        vehicle = _find_hero(world)
        if vehicle is not None:
            break
        if args.wait_timeout > 0 and time.monotonic() - start > args.wait_timeout:
            raise TimeoutError("Timed out waiting for a hero vehicle")
        time.sleep(0.2)

    if stopped:
        return

    assert vehicle is not None
    print(f"Recording actor {vehicle.id} ({vehicle.type_id}) -> {out_path}")

    frame_queue: queue.Queue = queue.Queue(maxsize=16)
    camera = _make_camera(world, carla, vehicle, args)
    camera.listen(lambda image: frame_queue.put(image) if not frame_queue.full() else None)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*args.codec),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        camera.stop()
        camera.destroy()
        raise RuntimeError(f"Could not open video writer for {out_path}")

    recorded = 0
    deadline = time.monotonic() + args.duration_sec if args.duration_sec > 0 else None
    no_frame_since = None

    try:
        while not stopped:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if not vehicle.is_alive:
                break
            try:
                image = frame_queue.get(timeout=1.0)
            except queue.Empty:
                if no_frame_since is None:
                    no_frame_since = time.monotonic()
                if time.monotonic() - no_frame_since > args.no_frame_timeout:
                    print("No frames received; stopping recorder.")
                    break
                continue
            no_frame_since = None
            writer.write(_image_to_bgr(image))
            recorded += 1
            if args.report_every > 0 and recorded % args.report_every == 0:
                print(f"recorded_frames={recorded} sim_frame={image.frame}")
    finally:
        camera.stop()
        camera.destroy()
        writer.release()

    print(f"Saved {recorded} frames to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--carla-root", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--view", choices=["chase", "topdown", "front"], default="chase")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--sensor-tick", type=float, default=0.05)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    parser.add_argument("--no-frame-timeout", type=float, default=10.0)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--report-every", type=int, default=200)
    record(parser.parse_args())


if __name__ == "__main__":
    main()
