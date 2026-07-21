import argparse
import queue
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _install_carla_path(carla_root: str) -> None:
    try:
        import carla  # noqa: F401

        return
    except Exception:
        pass
    root = Path(carla_root).expanduser()
    dist = root / "PythonAPI" / "carla" / "dist"
    py_major = sys.version_info.major
    candidates = [
        root / "PythonAPI" / "carla",
        *[
            path
            for path in sorted(dist.glob("carla-*.egg"))
            if f"py{py_major}" in path.name or f"cp{py_major}" in path.name
        ],
        *[
            path
            for path in sorted(dist.glob("carla-*.whl"))
            if f"cp{py_major}" in path.name or "py3" in path.name
        ],
    ]
    for path in reversed(candidates):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _camera_rgb(image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3][:, :, ::-1].copy()


def _make_camera_bp(world, width: int, height: int, fov: float, fps: float):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    bp.set_attribute("sensor_tick", str(1.0 / fps))
    return bp


def _latest_image(sensor_queue: queue.Queue, timeout: float = 2.0):
    item = sensor_queue.get(timeout=timeout)
    while True:
        try:
            item = sensor_queue.get_nowait()
        except queue.Empty:
            return item


def _latest_rgb_or_previous(
    sensor_queue: queue.Queue,
    previous: np.ndarray | None,
    *,
    timeout: float,
    name: str,
    frame_idx: int,
) -> np.ndarray | None:
    try:
        return _camera_rgb(_latest_image(sensor_queue, timeout=timeout))
    except queue.Empty:
        if previous is None:
            raise
        if frame_idx % 100 == 0:
            print(f"warning: reused previous {name} frame at video_frame={frame_idx + 1}")
        return previous


def _find_hero(world):
    vehicles = list(world.get_actors().filter("vehicle.*"))
    for vehicle in vehicles:
        if vehicle.attributes.get("role_name") == "hero":
            return vehicle
    return vehicles[0] if vehicles else None


def _resize_to(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def _put_label(rgb: np.ndarray, text: str) -> None:
    cv2.rectangle(rgb, (0, 0), (260, 34), (0, 0, 0), thickness=-1)
    cv2.putText(rgb, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def _parse_recorder_map(info: str) -> str | None:
    match = re.search(r"^Map:\s*(\S+)", info, re.MULTILINE)
    return match.group(1) if match else None


def _parse_recorder_duration(info: str) -> float | None:
    match = re.search(r"^Duration:\s*([0-9.]+)\s*seconds", info, re.MULTILINE)
    return float(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--carla-root", default="/home/byeongjae/carla-simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--front-width", type=int, default=1024)
    parser.add_argument("--front-height", type=int, default=512)
    parser.add_argument("--front-fov", type=float, default=110.0)
    parser.add_argument("--view-width", type=int, default=960)
    parser.add_argument("--view-height", type=int, default=540)
    parser.add_argument("--skip-load-world", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--map-name", default="")
    parser.add_argument("--recorder-duration", type=float, default=0.0)
    parser.add_argument("--sensor-timeout", type=float, default=10.0)
    args = parser.parse_args()

    _install_carla_path(args.carla_root)
    import carla

    record_path = str(Path(args.record).expanduser())
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    if args.map_name or args.recorder_duration > 0:
        map_name = args.map_name or None
        recorder_duration = args.recorder_duration if args.recorder_duration > 0 else None
    else:
        info = client.show_recorder_file_info(record_path, True)
        map_name = _parse_recorder_map(info)
        recorder_duration = _parse_recorder_duration(info)

    if map_name and not args.skip_load_world:
        client.load_world(map_name)
        time.sleep(1.0)

    world = client.get_world()
    original_settings = world.get_settings()
    actors = []
    writer = None
    try:
        if args.sync:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / args.fps
            world.apply_settings(settings)

        replay_duration = args.duration if args.duration > 0 else 0.0
        print(client.replay_file(record_path, args.start, replay_duration, 0))

        hero = None
        for _ in range(120):
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick(2.0)
            hero = _find_hero(world)
            if hero is not None:
                break
        if hero is None:
            raise RuntimeError("Could not find replayed hero vehicle")

        front_bp = _make_camera_bp(world, args.front_width, args.front_height, args.front_fov, args.fps)
        top_bp = _make_camera_bp(world, args.view_width, args.view_height, 90.0, args.fps)
        front_tf = carla.Transform(
            carla.Location(x=-1.5, y=0.0, z=2.0),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        top_tf = carla.Transform(
            carla.Location(x=-7.0, y=0.0, z=18.0),
            carla.Rotation(pitch=-72.0, yaw=0.0, roll=0.0),
        )

        front_q: queue.Queue = queue.Queue()
        top_q: queue.Queue = queue.Queue()
        front = world.spawn_actor(front_bp, front_tf, attach_to=hero)
        top = world.spawn_actor(top_bp, top_tf, attach_to=hero)
        actors.extend([front, top])
        front.listen(front_q.put)
        top.listen(top_q.put)

        duration = args.duration if args.duration > 0 else (recorder_duration or 0.0)
        if duration <= 0:
            duration = 60.0
        frame_count = max(1, int(round(duration * args.fps)))
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.view_width * 2, args.view_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output_path}")

        previous_front_rgb = None
        previous_top_rgb = None
        for frame_idx in range(frame_count):
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick(2.0)
            previous_front_rgb = _latest_rgb_or_previous(
                front_q,
                previous_front_rgb,
                timeout=args.sensor_timeout,
                name="front",
                frame_idx=frame_idx,
            )
            previous_top_rgb = _latest_rgb_or_previous(
                top_q,
                previous_top_rgb,
                timeout=args.sensor_timeout,
                name="topdown",
                frame_idx=frame_idx,
            )
            front_rgb = _resize_to(previous_front_rgb, args.view_width, args.view_height)
            top_rgb = _resize_to(previous_top_rgb, args.view_width, args.view_height)
            _put_label(front_rgb, "official TransFuser++ front")
            _put_label(top_rgb, "observer topdown")
            frame_rgb = np.concatenate([front_rgb, top_rgb], axis=1)
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            if (frame_idx + 1) % max(1, int(args.fps * 5)) == 0:
                print(f"video_frame={frame_idx + 1}/{frame_count}")
    finally:
        if writer is not None:
            writer.release()
        for actor in actors:
            try:
                actor.destroy()
            except Exception:
                pass
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

    print({"output": str(output_path), "map": map_name, "recorder_duration": recorder_duration})


if __name__ == "__main__":
    main()
