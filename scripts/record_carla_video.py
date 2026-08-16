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


VIEW_CHOICES = ["chase", "overhead", "close_topdown", "topdown", "front"]


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
    if view == "overhead":
        return carla.Transform(carla.Location(x=-5.5, y=0.0, z=8.0), carla.Rotation(pitch=-58.0))
    if view == "close_topdown":
        return carla.Transform(carla.Location(x=0.0, y=0.0, z=12.0), carla.Rotation(pitch=-90.0))
    if view == "front":
        return carla.Transform(carla.Location(x=1.6, y=0.0, z=2.2), carla.Rotation(pitch=-5.0))
    return carla.Transform(carla.Location(x=-8.0, y=0.0, z=3.2), carla.Rotation(pitch=-15.0))


def _make_camera(world, carla, vehicle, view: str, width: int, height: int, fov: float, sensor_tick: float):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(fov))
    if sensor_tick > 0:
        blueprint.set_attribute("sensor_tick", str(sensor_tick))
    return world.spawn_actor(blueprint, _view_transform(carla, view), attach_to=vehicle)


def _image_to_bgr(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3].copy()


def _put_latest(frame_queue: queue.Queue, image) -> None:
    try:
        frame_queue.put_nowait(image)
    except queue.Full:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            frame_queue.put_nowait(image)
        except queue.Full:
            pass


def _wait_for_image(world, frame_queue: queue.Queue, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return frame_queue.get(timeout=0.1)
        except queue.Empty:
            try:
                world.wait_for_tick(0.2)
            except Exception:
                time.sleep(0.05)
    raise queue.Empty


def _drain_latest_image(frame_queue: queue.Queue):
    image = None
    while True:
        try:
            image = frame_queue.get_nowait()
        except queue.Empty:
            return image


def _pip_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    scale = max(0.05, min(0.80, scale))
    pip_width = max(2, int(round(width * scale)))
    pip_height = max(2, int(round(height * scale)))
    return pip_width, pip_height


def _overlay_pip(frame: np.ndarray, pip_frame: np.ndarray, scale: float, margin: int, border: int) -> np.ndarray:
    height, width = frame.shape[:2]
    pip_width, pip_height = _pip_dimensions(width, height, scale)
    inset = cv2.resize(pip_frame, (pip_width, pip_height), interpolation=cv2.INTER_AREA)

    margin = max(0, margin)
    border = max(0, border)
    x0 = max(0, width - pip_width - margin)
    y0 = max(0, height - pip_height - margin)
    x1 = min(width, x0 + pip_width)
    y1 = min(height, y0 + pip_height)

    if border > 0:
        bx0 = max(0, x0 - border)
        by0 = max(0, y0 - border)
        bx1 = min(width, x1 + border)
        by1 = min(height, y1 + border)
        frame[by0:by1, bx0:bx1] = (245, 245, 245)

    frame[y0:y1, x0:x1] = inset[: y1 - y0, : x1 - x0]
    return frame


def record(args: argparse.Namespace) -> None:
    _install_carla_path(args.carla_root)
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

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
    world = client.get_world()
    start = time.monotonic()
    while not stopped:
        world = client.get_world()
        vehicle = _find_hero(world)
        if vehicle is not None:
            break
        if args.wait_timeout > 0 and time.monotonic() - start > args.wait_timeout:
            raise TimeoutError("Timed out waiting for a hero vehicle")
        time.sleep(0.2)

    if stopped:
        return

    assert vehicle is not None
    world = client.get_world()
    current_vehicle = world.get_actor(vehicle.id)
    if current_vehicle is not None:
        vehicle = current_vehicle
    settings = world.get_settings()
    print(
        "Recording actor "
        f"{vehicle.id} ({vehicle.type_id}) -> {out_path} "
        f"sync={settings.synchronous_mode} fixed_dt={settings.fixed_delta_seconds}",
        flush=True,
    )

    frame_queue: queue.Queue = queue.Queue(maxsize=64)
    camera = _make_camera(world, carla, vehicle, args.view, args.width, args.height, args.fov, args.sensor_tick)
    camera.listen(lambda image: _put_latest(frame_queue, image))

    pip_queue: queue.Queue | None = None
    pip_camera = None
    pip_view = args.pip_view.strip() if args.pip_view else ""
    latest_pip_frame = None
    if pip_view:
        pip_width, pip_height = _pip_dimensions(args.width, args.height, args.pip_scale)
        pip_queue = queue.Queue(maxsize=64)
        pip_camera = _make_camera(world, carla, vehicle, pip_view, pip_width, pip_height, args.fov, args.sensor_tick)
        pip_camera.listen(lambda image: _put_latest(pip_queue, image))
        print(
            f"Picture-in-picture enabled: main={args.view} pip={pip_view} "
            f"pip_size={pip_width}x{pip_height}",
            flush=True,
        )

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*args.codec),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        camera.stop()
        camera.destroy()
        if pip_camera is not None:
            pip_camera.stop()
            pip_camera.destroy()
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
                image = _wait_for_image(world, frame_queue, timeout=1.0)
            except queue.Empty:
                if no_frame_since is None:
                    no_frame_since = time.monotonic()
                if time.monotonic() - no_frame_since > args.no_frame_timeout:
                    print("No frames received; stopping recorder.")
                    break
                continue
            no_frame_since = None
            frame = _image_to_bgr(image)
            if pip_queue is not None:
                pip_image = _drain_latest_image(pip_queue)
                if pip_image is not None:
                    latest_pip_frame = _image_to_bgr(pip_image)
                if latest_pip_frame is not None:
                    frame = _overlay_pip(frame, latest_pip_frame, args.pip_scale, args.pip_margin, args.pip_border)
            writer.write(frame)
            recorded += 1
            if args.report_every > 0 and recorded % args.report_every == 0:
                print(f"recorded_frames={recorded} sim_frame={image.frame}", flush=True)
    finally:
        camera.stop()
        camera.destroy()
        if pip_camera is not None:
            pip_camera.stop()
            pip_camera.destroy()
        writer.release()

    print(f"Saved {recorded} frames to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--carla-root", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--view", choices=VIEW_CHOICES, default="chase")
    parser.add_argument("--pip-view", choices=VIEW_CHOICES, default="")
    parser.add_argument("--pip-scale", type=float, default=0.30)
    parser.add_argument("--pip-margin", type=int, default=24)
    parser.add_argument("--pip-border", type=int, default=2)
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
