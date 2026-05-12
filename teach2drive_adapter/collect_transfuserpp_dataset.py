"""Collect CARLA data with TransFuser++-style sensors.

This collector is intentionally independent from the CARLA leaderboard runner.
It creates Teach2Drive-friendly episode folders while also writing the main
TransFuser++ inference inputs: RGB, raw ego-frame LiDAR, LiDAR BEV, speed,
target point, route, command, controls, IMU, and sensor-layout metadata.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import queue
import random
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - keeps --help usable before deps are installed.
    np = None  # type: ignore[assignment]

try:
    import cv2
except ImportError:  # pragma: no cover - keeps --help usable before deps are installed.
    cv2 = None  # type: ignore[assignment]


LANE_FOLLOW_COMMAND = 4


@dataclass(frozen=True)
class Pose6D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class CameraRigSpec:
    pose: Pose6D
    width: int = 1024
    height: int = 512
    fov: float = 110.0


@dataclass(frozen=True)
class LidarRigSpec:
    pose: Pose6D
    channels: int = 64
    range: float = 85.0
    points_per_second: int = 600000
    rotation_frequency: float = 10.0
    upper_fov: float = 10.0
    lower_fov: float = -30.0


@dataclass(frozen=True)
class SensorProfile:
    name: str
    description: str
    cameras: Mapping[str, CameraRigSpec]
    lidar: LidarRigSpec


def _profile_tfpp_ego() -> SensorProfile:
    return SensorProfile(
        name="tfpp_ego",
        description="Official CARLA Garage TransFuser++ front RGB + roof LiDAR rig.",
        cameras={
            "front": CameraRigSpec(
                pose=Pose6D(x=-1.5, y=0.0, z=2.0, roll=0.0, pitch=0.0, yaw=0.0),
            ),
        },
        lidar=LidarRigSpec(
            pose=Pose6D(x=0.0, y=0.0, z=2.5, roll=0.0, pitch=0.0, yaw=-90.0),
        ),
    )


def _profile_front_triplet_shifted() -> SensorProfile:
    return SensorProfile(
        name="front_triplet_shifted",
        description=(
            "Three front-facing RGB cameras with small lateral/yaw offsets plus a "
            "slightly shifted LiDAR. Names stay left/front/right for adapter compatibility."
        ),
        cameras={
            "left": CameraRigSpec(
                pose=Pose6D(x=-1.45, y=-0.35, z=2.05, roll=0.0, pitch=0.0, yaw=-8.0),
            ),
            "front": CameraRigSpec(
                pose=Pose6D(x=-1.5, y=0.0, z=2.0, roll=0.0, pitch=0.0, yaw=0.0),
            ),
            "right": CameraRigSpec(
                pose=Pose6D(x=-1.45, y=0.35, z=2.05, roll=0.0, pitch=0.0, yaw=8.0),
            ),
        },
        lidar=LidarRigSpec(
            pose=Pose6D(x=0.15, y=0.0, z=2.45, roll=0.0, pitch=0.0, yaw=-90.0),
        ),
    )


SENSOR_PROFILES = {
    "tfpp_ego": _profile_tfpp_ego,
    "front_triplet_shifted": _profile_front_triplet_shifted,
}


def _install_carla_python_path(carla_root: str) -> None:
    try:
        import carla  # noqa: F401

        return
    except Exception:
        pass

    root = Path(carla_root).expanduser()
    dist = root / "PythonAPI" / "carla" / "dist"
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    candidates = [
        root / "PythonAPI" / "carla",
        *[path for path in sorted(dist.glob("carla-*.whl")) if f"cp{py_major}{py_minor}" in path.name],
        *[path for path in sorted(dist.glob("carla-*.egg")) if f"py{py_major}.{py_minor}" in path.name],
    ]
    for path in reversed(candidates):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_json_gz(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _require_runtime_deps() -> None:
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required for TransFuser++ data collection. "
            "Install this package or install teach2drive_adapter with its project dependencies."
        )
    if cv2 is None:
        raise ModuleNotFoundError(
            "opencv-python-headless is required for image writing. "
            "Install this package or install teach2drive_adapter with its project dependencies."
        )


def _camera_bgr(image) -> np.ndarray:
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    return array[:, :, :3].copy()


def _lidar_points(lidar) -> np.ndarray:
    points = np.frombuffer(lidar.raw_data, dtype=np.float32).reshape((-1, 4))
    return points.copy()


def _projected_speed(vehicle) -> float:
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    yaw = math.radians(float(transform.rotation.yaw))
    return float(velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw))


def _get_matching(sensor_queue: queue.Queue, frame: int, timeout: float):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            item = sensor_queue.get(timeout=max(0.01, deadline - time.monotonic()))
        except queue.Empty:
            break
        last = item
        if getattr(item, "frame", None) == frame:
            return item
    if last is not None:
        return last
    raise TimeoutError(f"Timed out waiting for sensor frame {frame}")


def _pose_to_transform(carla, pose: Pose6D):
    return carla.Transform(
        carla.Location(x=float(pose.x), y=float(pose.y), z=float(pose.z)),
        carla.Rotation(pitch=float(pose.pitch), yaw=float(pose.yaw), roll=float(pose.roll)),
    )


def _make_camera_bp(blueprints, name: str, spec: CameraRigSpec, hz: int):
    bp = blueprints.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(spec.width))
    bp.set_attribute("image_size_y", str(spec.height))
    bp.set_attribute("fov", str(spec.fov))
    bp.set_attribute("sensor_tick", str(1.0 / float(hz)))
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", f"tfpp_{name}")
    return bp


def _make_lidar_bp(blueprints, spec: LidarRigSpec, hz: int):
    bp = blueprints.find("sensor.lidar.ray_cast")
    bp.set_attribute("channels", str(spec.channels))
    bp.set_attribute("range", str(spec.range))
    bp.set_attribute("points_per_second", str(spec.points_per_second))
    bp.set_attribute("rotation_frequency", str(spec.rotation_frequency))
    bp.set_attribute("upper_fov", str(spec.upper_fov))
    bp.set_attribute("lower_fov", str(spec.lower_fov))
    bp.set_attribute("sensor_tick", str(1.0 / float(hz)))
    return bp


def _spawn_sensors(carla, world, vehicle, profile: SensorProfile, args, actors: List):
    blueprints = world.get_blueprint_library()
    camera_queues = {}
    for name, spec in profile.cameras.items():
        camera = world.spawn_actor(_make_camera_bp(blueprints, name, spec, args.hz), _pose_to_transform(carla, spec.pose), attach_to=vehicle)
        actors.append(camera)
        sensor_queue = queue.Queue()
        camera.listen(sensor_queue.put)
        camera_queues[name] = sensor_queue

    lidar = world.spawn_actor(_make_lidar_bp(blueprints, profile.lidar, args.hz), _pose_to_transform(carla, profile.lidar.pose), attach_to=vehicle)
    actors.append(lidar)
    lidar_q = queue.Queue()
    lidar.listen(lidar_q.put)

    imu_bp = blueprints.find("sensor.other.imu")
    imu_bp.set_attribute("sensor_tick", str(1.0 / float(args.hz)))
    imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
    actors.append(imu)
    imu_q = queue.Queue()
    imu.listen(imu_q.put)

    return camera_queues, lidar_q, imu_q


def _sensor_layout_payload(profile: SensorProfile) -> Mapping:
    return {
        "estimated": False,
        "profile": profile.name,
        "cameras": {
            name: {
                **asdict(spec.pose),
                "fov": spec.fov,
                "width": spec.width,
                "height": spec.height,
                "present": True,
            }
            for name, spec in profile.cameras.items()
        },
        "lidars": {
            "top": {
                **asdict(profile.lidar.pose),
                "channels": profile.lidar.channels,
                "range": profile.lidar.range,
                "present": True,
            }
        },
    }


def _transform_lidar_to_ego(points: np.ndarray, lidar_spec: LidarRigSpec) -> np.ndarray:
    yaw = math.radians(float(lidar_spec.pose.yaw))
    rotation = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    translation = np.asarray([lidar_spec.pose.x, lidar_spec.pose.y, lidar_spec.pose.z], dtype=np.float32)
    xyz = points[:, :3]
    return (rotation @ xyz.T).T + translation


def _align_lidar_to_current(previous_lidar: np.ndarray, translation: np.ndarray, yaw: float) -> np.ndarray:
    rotation = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (rotation.T @ (previous_lidar - translation).T).T


def _relative_motion_to_current(previous_transform, current_transform) -> Tuple[np.ndarray, float]:
    current_location = current_transform.location
    previous_location = previous_transform.location
    translation_world = np.asarray(
        [
            current_location.x - previous_location.x,
            current_location.y - previous_location.y,
            current_location.z - previous_location.z,
        ],
        dtype=np.float32,
    )
    current_yaw = math.radians(float(current_transform.rotation.yaw))
    rotation_world_to_current = np.asarray(
        [
            [math.cos(current_yaw), -math.sin(current_yaw), 0.0],
            [math.sin(current_yaw), math.cos(current_yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    translation_current = rotation_world_to_current.T @ translation_world
    yaw = math.radians(float(current_transform.rotation.yaw - previous_transform.rotation.yaw))
    return translation_current, yaw


def _lidar_bev(points: np.ndarray, args) -> np.ndarray:
    points = points[points[:, 2] < float(args.max_height_lidar)]
    points = points[points[:, 2] > float(args.min_height_lidar)]
    above = points[points[:, 2] > float(args.lidar_split_height)]
    xbins = np.linspace(args.min_x, args.max_x, int(round((args.max_x - args.min_x) * args.pixels_per_meter)) + 1)
    ybins = np.linspace(args.min_y, args.max_y, int(round((args.max_y - args.min_y) * args.pixels_per_meter)) + 1)
    hist = np.histogramdd(above[:, :2], bins=(xbins, ybins))[0] if len(above) else np.zeros((len(xbins) - 1, len(ybins) - 1))
    hist = np.minimum(hist, float(args.hist_max_per_pixel)) / float(args.hist_max_per_pixel)
    return hist.T[np.newaxis].astype(np.float32)


def _local_point(vehicle_transform, point_xy: Sequence[float]) -> List[float]:
    location = vehicle_transform.location
    yaw = math.radians(float(vehicle_transform.rotation.yaw))
    dx = float(point_xy[0]) - float(location.x)
    dy = float(point_xy[1]) - float(location.y)
    x = math.cos(yaw) * dx + math.sin(yaw) * dy
    y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return [float(x), float(y)]


def _route_points(carla_map, vehicle_transform, args) -> Tuple[List[List[float]], List[float], List[float]]:
    location = vehicle_transform.location
    waypoint = carla_map.get_waypoint(location, project_to_road=True)
    if waypoint is None:
        return [[float(args.target_point_distance_m), 0.0]], [float(args.target_point_distance_m), 0.0], [float(args.next_target_point_distance_m), 0.0]

    route = []
    current = waypoint
    travelled = 0.0
    target_world = None
    next_world = None
    step_m = max(float(args.route_step_m), 0.5)
    while len(route) < int(args.num_route_points):
        loc = current.transform.location
        local = _local_point(vehicle_transform, [loc.x, loc.y])
        route.append(local)
        if target_world is None and travelled >= float(args.target_point_distance_m):
            target_world = [loc.x, loc.y]
        if next_world is None and travelled >= float(args.next_target_point_distance_m):
            next_world = [loc.x, loc.y]
        next_points = current.next(step_m)
        if not next_points:
            break
        current = next_points[0]
        travelled += step_m

    if not route:
        route = [[float(args.target_point_distance_m), 0.0]]
    if target_world is None:
        target = route[min(len(route) - 1, 3)]
    else:
        target = _local_point(vehicle_transform, target_world)
    if next_world is None:
        target_next = route[-1]
    else:
        target_next = _local_point(vehicle_transform, next_world)
    return route, target, target_next


def _angle_from_target_point(target_point: Sequence[float]) -> float:
    return float(-math.degrees(math.atan2(-float(target_point[1]), max(float(target_point[0]), 1e-3))) / 90.0)


def _vehicle_light_state(vehicle) -> str:
    try:
        traffic_light = vehicle.get_traffic_light()
        if traffic_light is None:
            return "None"
        return str(vehicle.get_traffic_light_state()).rsplit(".", 1)[-1]
    except RuntimeError:
        return "Unknown"


def _hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if target.exists():
            target.unlink()
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _save_lidar_npz(path: Path, lidar_ego: np.ndarray, raw_points: np.ndarray, frame: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=lidar_ego.astype(np.float32), raw=raw_points.astype(np.float32), frame=np.asarray([frame], dtype=np.int64))


def _maybe_save_lidar_laz(path: Path, lidar_ego: np.ndarray) -> bool:
    try:
        import laspy
    except Exception:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3)
    header.offsets = np.min(lidar_ego, axis=0) if len(lidar_ego) else np.zeros(3, dtype=np.float32)
    header.scales = np.asarray([0.01, 0.01, 0.01], dtype=np.float64)
    with laspy.open(path, mode="w", header=header) as writer:
        record = laspy.ScaleAwarePointRecord.zeros(lidar_ego.shape[0], header=header)
        record.x = lidar_ego[:, 0]
        record.y = lidar_ego[:, 1]
        record.z = lidar_ego[:, 2]
        writer.write_points(record)
    return True


def _destroy_actors(client, actors: Iterable) -> None:
    actors = [actor for actor in actors if actor is not None]
    if not actors:
        return
    for actor in reversed(actors):
        try:
            if hasattr(actor, "is_alive") and not actor.is_alive:
                continue
            if hasattr(actor, "stop"):
                actor.stop()
        except RuntimeError:
            pass
    for actor in reversed(actors):
        try:
            if hasattr(actor, "is_alive") and not actor.is_alive:
                continue
            actor.destroy()
        except RuntimeError:
            pass


def _spawn_background_traffic(carla, client, world, traffic_manager, count: int, seed: int) -> List:
    if count <= 0:
        return []
    rng = random.Random(seed)
    blueprints = [bp for bp in world.get_blueprint_library().filter("vehicle.*") if bp.has_attribute("number_of_wheels")]
    spawn_points = list(world.get_map().get_spawn_points())
    rng.shuffle(spawn_points)
    actors = []
    for spawn in spawn_points[:count]:
        bp = rng.choice(blueprints)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "tfpp_background")
        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", rng.choice(colors))
        vehicle = world.try_spawn_actor(bp, spawn)
        if vehicle is None:
            continue
        vehicle.set_autopilot(True, traffic_manager.get_port())
        actors.append(vehicle)
    return actors


def _prepare_episode_dir(root: Path, profile: SensorProfile, episode_index: int, overwrite: bool) -> Path:
    profile_root = root / profile.name
    episode_dir = profile_root / f"episode_{episode_index:06d}"
    if episode_dir.exists() and not overwrite:
        raise FileExistsError(f"Episode directory already exists: {episode_dir}")
    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    for name in profile.cameras:
        (episode_dir / "camera" / name).mkdir(parents=True, exist_ok=True)
    (episode_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (episode_dir / "lidar").mkdir(parents=True, exist_ok=True)
    (episode_dir / "lidar_bev").mkdir(parents=True, exist_ok=True)
    (episode_dir / "measurements").mkdir(parents=True, exist_ok=True)
    return episode_dir


def _spawn_candidates(world, episode_index: int, seed: int, spawn_indices: Sequence[int]):
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("Current CARLA map has no spawn points")
    indices = list(range(len(spawn_points)))
    if spawn_indices:
        first = spawn_indices[episode_index % len(spawn_indices)] % len(spawn_points)
        indices.remove(first)
        indices.insert(0, first)
        return [spawn_points[index] for index in indices]
    rng = random.Random(seed + episode_index)
    rng.shuffle(indices)
    return [spawn_points[index] for index in indices]


def _spawn_ego_vehicle(world, vehicle_bp, episode_index: int, seed: int, spawn_indices: Sequence[int]):
    for spawn in _spawn_candidates(world, episode_index, seed, spawn_indices):
        vehicle = world.try_spawn_actor(vehicle_bp, spawn)
        if vehicle is not None:
            return vehicle, spawn
    raise RuntimeError("Could not spawn ego vehicle after trying every map spawn point. Try lowering --traffic-vehicles.")


def _episode_seconds_for(profile_index: int, episode_index: int, plan: Mapping[Tuple[int, int], float]) -> float:
    return float(plan[(profile_index, episode_index)])


def _build_episode_plan(profile_count: int, args) -> Dict[Tuple[int, int], float]:
    if args.episodes_per_profile > 0:
        return {
            (profile_index, episode_index): float(args.episode_sec)
            for profile_index in range(profile_count)
            for episode_index in range(args.episodes_per_profile)
        }

    total_seconds = float(args.duration_hours) * 3600.0
    if total_seconds <= 0:
        raise ValueError("--duration-hours must be positive when --episodes-per-profile is not set")
    per_profile = total_seconds / max(profile_count, 1)
    episodes = max(1, int(math.ceil(per_profile / float(args.episode_sec))))
    plan: Dict[Tuple[int, int], float] = {}
    for profile_index in range(profile_count):
        remaining = per_profile
        for episode_index in range(episodes):
            seconds = min(float(args.episode_sec), remaining)
            if seconds <= 0:
                break
            plan[(profile_index, episode_index)] = seconds
            remaining -= seconds
    return plan


def collect_episode(carla, client, world, traffic_manager, profile: SensorProfile, profile_index: int, episode_index: int, episode_sec: float, args) -> Mapping:
    episode_dir = _prepare_episode_dir(Path(args.output_root).expanduser(), profile, episode_index, args.overwrite)
    actors = []
    frame_records_path = episode_dir / "frames.jsonl"
    frame_records = frame_records_path.open("w", encoding="utf-8", buffering=1)
    started_wall = time.monotonic()
    bytes_written = 0
    saved_frames = 0
    last_lidar_ego = None
    last_vehicle_transform = None
    try:
        blueprints = world.get_blueprint_library()
        vehicle_bp = blueprints.filter(args.vehicle_filter)[0]
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "hero")
        vehicle, spawn = _spawn_ego_vehicle(world, vehicle_bp, episode_index + profile_index * 100000, args.seed, args.spawn_indices)
        actors.append(vehicle)
        vehicle.set_autopilot(False, traffic_manager.get_port())
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        traffic_manager.ignore_lights_percentage(vehicle, float(args.ignore_lights_percent))

        camera_queues, lidar_q, imu_q = _spawn_sensors(carla, world, vehicle, profile, args, actors)

        _write_json(episode_dir / "sensor_layout.json", _sensor_layout_payload(profile))
        episode_meta = {
            "dataset": "teach2drive_transfuserpp_carla",
            "profile": profile.name,
            "description": profile.description,
            "episode_index": episode_index,
            "episode_token": uuid.uuid4().hex,
            "map": world.get_map().name,
            "hz": args.hz,
            "save_every_n": args.save_every_n,
            "saved_fps": float(args.hz) / float(args.save_every_n),
            "episode_sec": episode_sec,
            "spawn_transform": {
                "location": [spawn.location.x, spawn.location.y, spawn.location.z],
                "rotation": [spawn.rotation.pitch, spawn.rotation.yaw, spawn.rotation.roll],
            },
            "sensor_profile": {
                "name": profile.name,
                "cameras": {name: asdict(spec) for name, spec in profile.cameras.items()},
                "lidar": asdict(profile.lidar),
            },
        }
        _write_json(episode_dir / "episode_meta.json", episode_meta)

        warmup_ticks = max(0, int(round(float(args.warmup_sec) * float(args.hz))))
        for _ in range(warmup_ticks):
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            world.tick()
        vehicle.set_autopilot(True, traffic_manager.get_port())

        ticks = max(1, int(round(float(episode_sec) * float(args.hz))))
        carla_map = world.get_map()
        start_elapsed = None
        for tick_index in range(ticks):
            frame = world.tick()
            snapshot = world.get_snapshot()
            if start_elapsed is None:
                start_elapsed = float(snapshot.timestamp.elapsed_seconds)

            camera_images = {name: _get_matching(sensor_queue, frame, args.sensor_timeout) for name, sensor_queue in camera_queues.items()}
            lidar_raw = _get_matching(lidar_q, frame, args.sensor_timeout)
            imu_data = _get_matching(imu_q, frame, args.sensor_timeout)
            raw_points = _lidar_points(lidar_raw)
            vehicle_transform = vehicle.get_transform()
            lidar_ego = _transform_lidar_to_ego(raw_points, profile.lidar)
            if last_lidar_ego is not None and last_vehicle_transform is not None:
                translation, yaw = _relative_motion_to_current(last_vehicle_transform, vehicle_transform)
                lidar_combined = np.concatenate([lidar_ego, _align_lidar_to_current(last_lidar_ego, translation, yaw)], axis=0)
            else:
                lidar_combined = lidar_ego
            last_lidar_ego = lidar_ego
            last_vehicle_transform = vehicle_transform

            if tick_index % int(args.save_every_n) != 0:
                continue

            control = vehicle.get_control()
            speed_mps = _projected_speed(vehicle)
            route, target_point, target_point_next = _route_points(carla_map, vehicle_transform, args)
            speed_limit_mps = float(vehicle.get_speed_limit()) / 3.6
            target_speed = float(args.target_speed_mps) if args.target_speed_mps > 0 else speed_limit_mps
            frame_id = saved_frames
            frame_token = uuid.uuid4().hex

            camera_tokens = {}
            for name, image in camera_images.items():
                image_path = episode_dir / "camera" / name / f"{frame_id:04d}.jpg"
                ok = cv2.imwrite(str(image_path), _camera_bgr(image), [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
                if not ok:
                    raise RuntimeError(f"Failed to write camera image: {image_path}")
                bytes_written += image_path.stat().st_size
                camera_tokens[name] = str(image_path.relative_to(episode_dir))
            if "front" in camera_tokens:
                _hardlink_or_copy(episode_dir / camera_tokens["front"], episode_dir / "rgb" / f"{frame_id:04d}.jpg")

            lidar_npz_path = episode_dir / "lidar" / f"{frame_id:04d}.npz"
            _save_lidar_npz(lidar_npz_path, lidar_combined, raw_points, frame)
            lidar_laz_path = None
            if args.lidar_format in {"laz", "both"}:
                candidate = episode_dir / "lidar" / f"{frame_id:04d}.laz"
                if _maybe_save_lidar_laz(candidate, lidar_combined):
                    lidar_laz_path = candidate
            bytes_written += lidar_npz_path.stat().st_size
            if lidar_laz_path is not None:
                bytes_written += lidar_laz_path.stat().st_size

            bev = _lidar_bev(lidar_combined, args)
            lidar_bev_path = episode_dir / "lidar_bev" / f"{frame_id:04d}.npy"
            np.save(lidar_bev_path, bev.astype(np.float16))
            bytes_written += lidar_bev_path.stat().st_size

            location = vehicle_transform.location
            rotation = vehicle_transform.rotation
            velocity = vehicle.get_velocity()
            angular = vehicle.get_angular_velocity()
            measurement = {
                "pos_global": [float(location.x), float(location.y)],
                "z": float(location.z),
                "theta": math.radians(float(rotation.yaw)),
                "speed": speed_mps,
                "target_speed": target_speed,
                "speed_limit": speed_limit_mps,
                "target_point": target_point,
                "target_point_next": target_point_next,
                "command": LANE_FOLLOW_COMMAND,
                "next_command": LANE_FOLLOW_COMMAND,
                "aim_wp": target_point,
                "route": route,
                "route_original": route,
                "changed_route": False,
                "steer": float(control.steer),
                "throttle": float(control.throttle),
                "brake": bool(control.brake > 0.05),
                "brake_float": float(control.brake),
                "control_brake": bool(control.brake > 0.05),
                "junction": False,
                "vehicle_hazard": False,
                "vehicle_affecting_id": None,
                "light_hazard": _vehicle_light_state(vehicle) == "Red",
                "walker_hazard": False,
                "walker_affecting_id": None,
                "stop_sign_hazard": False,
                "stop_sign_close": False,
                "walker_close": False,
                "walker_close_id": None,
                "angle": _angle_from_target_point(target_point),
                "augmentation_translation": 0.0,
                "augmentation_rotation": 0.0,
                "ego_matrix": vehicle_transform.get_matrix(),
                "velocity": [float(velocity.x), float(velocity.y), float(velocity.z)],
                "angular_velocity": [float(angular.x), float(angular.y), float(angular.z)],
                "imu": {
                    "accelerometer": [float(imu_data.accelerometer.x), float(imu_data.accelerometer.y), float(imu_data.accelerometer.z)],
                    "gyroscope": [float(imu_data.gyroscope.x), float(imu_data.gyroscope.y), float(imu_data.gyroscope.z)],
                    "compass": float(getattr(imu_data, "compass", float("nan"))),
                },
            }
            measurement_path = episode_dir / "measurements" / f"{frame_id:04d}.json.gz"
            _write_json_gz(measurement_path, measurement)
            bytes_written += measurement_path.stat().st_size

            record = {
                "episode_token": episode_meta["episode_token"],
                "frame_token": frame_token,
                "step": frame_id,
                "carla_tick": tick_index,
                "carla_frame": int(frame),
                "time": float(snapshot.timestamp.elapsed_seconds - start_elapsed),
                "profile": profile.name,
                "camera_tokens": camera_tokens,
                "rgb_token": f"rgb/{frame_id:04d}.jpg" if "front" in camera_tokens else None,
                "lidar_token": str(lidar_npz_path.relative_to(episode_dir)),
                "lidar_laz_token": str(lidar_laz_path.relative_to(episode_dir)) if lidar_laz_path is not None else None,
                "lidar_bev_token": str(lidar_bev_path.relative_to(episode_dir)),
                "measurement_token": str(measurement_path.relative_to(episode_dir)),
                "odom": {
                    "x": float(location.x),
                    "y": float(location.y),
                    "z": float(location.z),
                    "roll": math.radians(float(rotation.roll)),
                    "pitch": math.radians(float(rotation.pitch)),
                    "yaw": math.radians(float(rotation.yaw)),
                    "v_forward": speed_mps,
                },
                "control": {
                    "throttle": float(control.throttle),
                    "steer": float(control.steer),
                    "brake": float(control.brake),
                },
            }
            frame_records.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved_frames += 1

            if saved_frames % max(1, int(args.report_every_frames)) == 0:
                wall = time.monotonic() - started_wall
                mib = bytes_written / (1024 * 1024)
                print(
                    f"profile={profile.name} episode={episode_index} saved={saved_frames} "
                    f"sim={tick_index / float(args.hz):.1f}/{episode_sec:.1f}s "
                    f"written={mib:.1f}MiB wall_rate={mib / max(wall, 1e-6):.2f}MiB/s",
                    flush=True,
                )

        summary = {
            "profile": profile.name,
            "episode_index": episode_index,
            "episode_sec": episode_sec,
            "saved_frames": saved_frames,
            "bytes_written": int(bytes_written),
            "elapsed_wall_sec": time.monotonic() - started_wall,
            "episode_dir": str(episode_dir),
        }
        _write_json(episode_dir / "episode_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return summary
    finally:
        frame_records.close()
        _destroy_actors(client, actors)


def collect(args) -> None:
    _require_runtime_deps()
    _install_carla_python_path(args.carla_root)
    import carla

    profile_names = [name.strip() for name in args.profiles.split(",") if name.strip()]
    unknown = [name for name in profile_names if name not in SENSOR_PROFILES]
    if unknown:
        raise ValueError(f"Unknown profiles {unknown}; choose from {sorted(SENSOR_PROFILES)}")
    profiles = [SENSOR_PROFILES[name]() for name in profile_names]

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(float(args.timeout))
    world = client.get_world()
    if args.map:
        short = str(args.map).split("/")[-1]
        if short not in world.get_map().name:
            world = client.load_world(short)
            time.sleep(1.0)

    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(int(args.traffic_manager_port))
    background_actors = []
    summaries = []
    episode_plan = _build_episode_plan(len(profiles), args)
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / float(args.hz)
        settings.no_rendering_mode = bool(args.no_rendering)
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(int(args.seed))
        traffic_manager.set_global_distance_to_leading_vehicle(float(args.global_distance_to_leading_vehicle))
        traffic_manager.global_percentage_speed_difference(float(args.global_speed_difference))

        background_actors = _spawn_background_traffic(carla, client, world, traffic_manager, int(args.traffic_vehicles), int(args.seed) + 1701)

        dataset_meta = {
            "dataset": "teach2drive_transfuserpp_carla",
            "profiles": [profile.name for profile in profiles],
            "map": world.get_map().name,
            "hz": args.hz,
            "save_every_n": args.save_every_n,
            "saved_fps": float(args.hz) / float(args.save_every_n),
            "camera_width": 1024,
            "camera_height": 512,
            "camera_fov": 110.0,
            "lidar_points_per_second": 600000,
            "lidar_rotation_frequency": 10.0,
            "lidar_bev": {
                "shape": [1, int((args.max_x - args.min_x) * args.pixels_per_meter), int((args.max_y - args.min_y) * args.pixels_per_meter)],
                "min_x": args.min_x,
                "max_x": args.max_x,
                "min_y": args.min_y,
                "max_y": args.max_y,
                "pixels_per_meter": args.pixels_per_meter,
                "use_ground_plane": False,
            },
            "transfuserpp_measurements": [
                "speed",
                "target_speed",
                "target_point",
                "target_point_next",
                "route",
                "command",
                "next_command",
                "theta",
                "controls",
            ],
        }
        _write_json(output_root / "dataset_meta.json", dataset_meta)

        for profile_index, profile in enumerate(profiles):
            episode_indices = sorted(index for pidx, index in episode_plan if pidx == profile_index)
            for episode_index in episode_indices:
                episode_sec = _episode_seconds_for(profile_index, episode_index, episode_plan)
                summaries.append(collect_episode(carla, client, world, traffic_manager, profile, profile_index, episode_index, episode_sec, args))
                _write_json(output_root / "dataset_summary.json", {"episodes": summaries})
    finally:
        _destroy_actors(client, background_actors)
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


def _parse_spawn_indices(value: str) -> List[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect TransFuser++-style CARLA data for Teach2Drive layout adaptation.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--carla-root", default="/home/jovyan/dataset/byeongjae/carla-simulator")
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--profiles", default="tfpp_ego,front_triplet_shifted")
    parser.add_argument("--duration-hours", type=float, default=2.5, help="Total simulated saved duration across all profiles.")
    parser.add_argument("--episode-sec", type=float, default=300.0)
    parser.add_argument("--episodes-per-profile", type=int, default=0, help="If set, ignore --duration-hours and collect this many episodes per profile.")
    parser.add_argument("--hz", type=int, default=20)
    parser.add_argument("--save-every-n", type=int, default=5, help="Official TransFuser++ data_save_freq is 5 at 20 Hz, i.e. 4 saved FPS.")
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--sensor-timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spawn-indices", type=_parse_spawn_indices, default=[])
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--traffic-vehicles", type=int, default=60)
    parser.add_argument("--global-distance-to-leading-vehicle", type=float, default=2.5)
    parser.add_argument("--global-speed-difference", type=float, default=0.0)
    parser.add_argument("--ignore-lights-percent", type=float, default=0.0)
    parser.add_argument("--target-speed-mps", type=float, default=0.0)
    parser.add_argument("--target-point-distance-m", type=float, default=12.0)
    parser.add_argument("--next-target-point-distance-m", type=float, default=24.0)
    parser.add_argument("--num-route-points", type=int, default=20)
    parser.add_argument("--route-step-m", type=float, default=2.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--lidar-format", choices=["npz", "laz", "both"], default="npz")
    parser.add_argument("--min-x", type=float, default=-32.0)
    parser.add_argument("--max-x", type=float, default=32.0)
    parser.add_argument("--min-y", type=float, default=-32.0)
    parser.add_argument("--max-y", type=float, default=32.0)
    parser.add_argument("--min-height-lidar", type=float, default=-100.0)
    parser.add_argument("--max-height-lidar", type=float, default=100.0)
    parser.add_argument("--lidar-split-height", type=float, default=0.2)
    parser.add_argument("--pixels-per-meter", type=float, default=4.0)
    parser.add_argument("--hist-max-per-pixel", type=float, default=5.0)
    parser.add_argument("--report-every-frames", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-rendering", action="store_true")
    return parser


def main() -> None:
    collect(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
