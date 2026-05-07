import argparse
import json
import math
import queue
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch


def _install_bootstrap_path() -> None:
    candidates = []
    env_root = None
    try:
        import os

        env_root = os.environ.get("TEACH2DRIVE_BOOTSTRAP_ROOT")
    except Exception:
        env_root = None
    if env_root:
        candidates.append(Path(env_root).expanduser())
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[2] / "teach2drive_bootstrap",
            here.parents[1] / "teach2drive_bootstrap",
            Path.home() / "teach2drive" / "workspace" / "teach2drive_bootstrap",
        ]
    )
    for root in candidates:
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
            return


def _install_carla_python_path() -> None:
    try:
        import carla  # noqa: F401

        return
    except Exception:
        pass
    try:
        import os

        carla_root = os.environ.get("CARLA_ROOT")
    except Exception:
        carla_root = None
    if not carla_root:
        return
    root = Path(carla_root).expanduser()
    dist = root / "PythonAPI" / "carla" / "dist"
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    eggs = [path for path in sorted(dist.glob("carla-*.egg")) if f"py{py_major}" in path.name]
    wheels = [path for path in sorted(dist.glob("carla-*.whl")) if f"cp{py_major}{py_minor}" in path.name or "py3" in path.name]
    candidates = [*eggs, *wheels, root / "PythonAPI" / "carla"]
    seen = set()
    for path in reversed(candidates):
        if path in seen:
            continue
        seen.add(path)
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_install_bootstrap_path()
_install_carla_python_path()

from teach2drive.carla_collect import _carla_image_to_rgb, _carla_lidar_to_bev, _destroy_actors, _get_matching, _import_carla
from teach2drive.carla_collect_tokens import CAMERA_TRANSFORMS
from teach2drive.carla_rollout import (
    _camera_blueprint,
    _collision_key,
    _location_text,
    _new_infraction_log,
    _open_video_writer,
    _render_video_frame,
    _score_like_leaderboard,
    _short_map_name,
)
from teach2drive.carla_rollout_tokens import (
    _apply_control,
    _load_token_route,
    _make_scalar_monotonic,
    _projected_speed,
)

from .sensor_layout import flatten_sensor_layout, load_sensor_layout, teach2drive_tokens_layout
from .train_transfuserpp_cached_adapter import CachedTransFuserPPAdapterPolicy
from .train_transfuserpp_cached_visual_adapter import CachedVisualTransFuserPPAdapterPolicy
from .transfuserpp_bridge import base_target_from_checkpoint, load_transfuserpp, prepare_transfuserpp_inputs, speed_expectation


def _camera_list(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _camera_transform(carla, name):
    location, rotation = CAMERA_TRANSFORMS[name]
    return carla.Transform(
        carla.Location(x=location[0], y=location[1], z=location[2]),
        carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2]),
    )


def _spawn_cameras(carla, world, blueprints, vehicle, cameras, args, actors):
    queues = {}
    for name in cameras:
        camera_bp = _camera_blueprint(blueprints, args.image_size, args.camera_fov, args.hz)
        sensor = world.spawn_actor(camera_bp, _camera_transform(carla, name), attach_to=vehicle)
        actors.append(sensor)
        sensor_q = queue.Queue()
        sensor.listen(sensor_q.put)
        queues[name] = sensor_q
    return queues


def _spawn_lidar(carla, world, blueprints, vehicle, args, actors):
    lidar_bp = blueprints.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", str(args.lidar_channels))
    lidar_bp.set_attribute("range", str(args.lidar_range))
    lidar_bp.set_attribute("points_per_second", str(args.lidar_points_per_second))
    lidar_bp.set_attribute("rotation_frequency", str(args.hz))
    lidar_bp.set_attribute("upper_fov", str(args.lidar_upper_fov))
    lidar_bp.set_attribute("lower_fov", str(args.lidar_lower_fov))
    lidar_bp.set_attribute("sensor_tick", str(1.0 / args.hz))
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=args.lidar_z)), attach_to=vehicle)
    actors.append(lidar)
    lidar_q = queue.Queue()
    lidar.listen(lidar_q.put)
    return lidar_q


def _spawn_imu(carla, world, blueprints, vehicle, args, actors):
    imu_bp = blueprints.find("sensor.other.imu")
    imu_bp.set_attribute("sensor_tick", str(1.0 / args.hz))
    imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
    actors.append(imu)
    imu_q = queue.Queue()
    imu.listen(imu_q.put)
    return imu_q


def _checkpoint_has_visual_adapter(state: Dict[str, torch.Tensor]) -> bool:
    return any(str(key).startswith("visual.") for key in state)


def _resolve_model_paths(args, checkpoint):
    ckpt_args = checkpoint.get("args", {})
    metadata = checkpoint.get("cache_metadata", {})
    garage_root = args.garage_root or str(metadata.get("garage_root", "")) or str(ckpt_args.get("garage_root", ""))
    team_config = args.team_config or str(metadata.get("team_config", "")) or str(ckpt_args.get("team_config", ""))
    tfpp_checkpoint = args.tfpp_checkpoint
    if not garage_root:
        raise ValueError("--garage-root is required because the checkpoint does not contain cache metadata")
    if not team_config:
        raise ValueError("--team-config is required because the checkpoint does not contain cache metadata")
    return garage_root, team_config, tfpp_checkpoint


def _load_cached_adapter(args, device):
    checkpoint_path = Path(args.checkpoint).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model_state"]
    ckpt_args = checkpoint.get("args", {})
    metadata = checkpoint.get("cache_metadata", {})
    is_visual = _checkpoint_has_visual_adapter(state)

    garage_root, team_config, tfpp_checkpoint = _resolve_model_paths(args, checkpoint)
    prior_net, prior_config, prior_load_info = load_transfuserpp(garage_root, team_config, device=device, checkpoint=tfpp_checkpoint)

    target_dim = int(checkpoint.get("target_dim", 17))
    speed_dim = int(ckpt_args.get("speed_dim", args.speed_dim))
    checkpoint_dim = int(checkpoint.get("checkpoint_dim", 20))
    speed_classes = int(checkpoint.get("speed_classes", len(getattr(prior_config, "target_speeds", [])) or 1))
    scalar_dim = int(checkpoint.get("scalar_dim", 20))
    layout_dim = int(checkpoint.get("layout_dim", 52))

    if is_visual:
        cameras = _camera_list(checkpoint.get("cameras") or ckpt_args.get("cameras") or "front,left,right")
        model = CachedVisualTransFuserPPAdapterPolicy(
            scalar_dim=scalar_dim,
            layout_dim=layout_dim,
            target_dim=target_dim,
            checkpoint_dim=checkpoint_dim,
            speed_classes=speed_classes,
            cameras=cameras,
            lidar_channels=int(checkpoint.get("lidar_channels", ckpt_args.get("lidar_channels", args.visual_lidar_channels))),
            hidden_dim=int(ckpt_args.get("hidden_dim", args.hidden_dim)),
            layout_hidden_dim=int(ckpt_args.get("layout_hidden_dim", args.layout_hidden_dim)),
            visual_dim=int(ckpt_args.get("visual_dim", args.visual_dim)),
            visual_token_dim=int(ckpt_args.get("visual_token_dim", args.visual_token_dim)),
            visual_layers=int(ckpt_args.get("visual_layers", args.visual_layers)),
            visual_heads=int(ckpt_args.get("visual_heads", args.visual_heads)),
        )
        model_mode = "transfuserpp_cached_visual_bev_adapter"
        visual_size = tuple(int(v) for v in ckpt_args.get("image_size", args.visual_image_size))
        visual_lidar_size = int(ckpt_args.get("lidar_size", args.visual_lidar_size))
    else:
        cameras = _camera_list(args.cameras or metadata.get("cameras") or ckpt_args.get("cameras") or "front")
        model = CachedTransFuserPPAdapterPolicy(
            scalar_dim=scalar_dim,
            layout_dim=layout_dim,
            target_dim=target_dim,
            checkpoint_dim=checkpoint_dim,
            speed_classes=speed_classes,
            hidden_dim=int(ckpt_args.get("hidden_dim", args.hidden_dim)),
            layout_hidden_dim=int(ckpt_args.get("layout_hidden_dim", args.layout_hidden_dim)),
        )
        model_mode = "transfuserpp_cached_residual_adapter"
        visual_size = tuple(int(v) for v in args.visual_image_size)
        visual_lidar_size = int(args.visual_lidar_size)

    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    prior_net.eval()

    tfpp_camera = args.tfpp_camera or str(metadata.get("tfpp_camera", "")) or "front"
    command_mode = args.command_mode or str(metadata.get("command_mode", "")) or "target_angle"
    spawn_cameras = list(dict.fromkeys([*cameras, tfpp_camera, "front"]))
    bad = [name for name in spawn_cameras if name not in CAMERA_TRANSFORMS]
    if bad:
        raise ValueError(f"Unknown camera names in checkpoint/args: {bad}")
    print(
        json.dumps(
            {
                "loaded_checkpoint": str(checkpoint_path),
                "model_mode": model_mode,
                "adapter_missing": len(missing),
                "adapter_unexpected": len(unexpected),
                "cameras": cameras,
                "spawn_cameras": spawn_cameras,
                "tfpp_camera": tfpp_camera,
                "command_mode": command_mode,
                "target_dim": target_dim,
                "speed_dim": speed_dim,
                "checkpoint_dim": checkpoint_dim,
                "speed_classes": speed_classes,
                "visual_image_size": list(visual_size),
                "visual_lidar_size": visual_lidar_size,
                "transfuserpp_load_info": prior_load_info,
            },
            indent=2,
        ),
        flush=True,
    )
    return {
        "adapter": model,
        "prior_net": prior_net,
        "prior_config": prior_config,
        "prior_load_info": prior_load_info,
        "mode": model_mode,
        "is_visual": is_visual,
        "cameras": cameras,
        "spawn_cameras": spawn_cameras,
        "tfpp_camera": tfpp_camera,
        "command_mode": command_mode,
        "target_dim": target_dim,
        "speed_dim": speed_dim,
        "checkpoint_dim": checkpoint_dim,
        "speed_classes": speed_classes,
        "layout_dim": layout_dim,
        "visual_size": visual_size,
        "visual_lidar_size": visual_lidar_size,
    }


def _flatten_checkpoint(pred_checkpoint: torch.Tensor, width: int) -> torch.Tensor:
    flat = torch.zeros((pred_checkpoint.shape[0], width), dtype=pred_checkpoint.dtype, device=pred_checkpoint.device)
    raw = pred_checkpoint.reshape(pred_checkpoint.shape[0], -1)
    flat[:, : min(width, raw.shape[1])] = raw[:, : flat.shape[1]]
    return flat


def _resize_rgb(image: np.ndarray, size: Sequence[int]) -> np.ndarray:
    size = tuple(int(v) for v in size)
    if (image.shape[1], image.shape[0]) == size:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _resize_lidar(bev: np.ndarray, size: int) -> np.ndarray:
    size = int(size)
    if bev.shape[-2:] == (size, size):
        return bev.astype(np.float32)
    channels = [cv2.resize(channel.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) for channel in bev]
    return np.stack(channels, axis=0).astype(np.float32)


def _match_lidar_channels(lidar_t: torch.Tensor, channels: int) -> torch.Tensor:
    if lidar_t.shape[1] == channels:
        return lidar_t
    if lidar_t.shape[1] > channels:
        return lidar_t[:, :channels]
    pad = torch.zeros(
        (lidar_t.shape[0], channels - lidar_t.shape[1], lidar_t.shape[2], lidar_t.shape[3]),
        dtype=lidar_t.dtype,
        device=lidar_t.device,
    )
    return torch.cat([lidar_t, pad], dim=1)


def _episode_layout_vector(episode_dir: Path, layout_dim: int) -> np.ndarray:
    try:
        layout = flatten_sensor_layout(load_sensor_layout(episode_dir))
    except Exception:
        layout = flatten_sensor_layout(teach2drive_tokens_layout())
    if layout.shape[0] == layout_dim:
        return layout.astype(np.float32)
    fixed = np.zeros((layout_dim,), dtype=np.float32)
    fixed[: min(layout_dim, layout.shape[0])] = layout[: min(layout_dim, layout.shape[0])]
    return fixed


def _predict(model_info, device, scalar, layout, image_by_name, lidar_bev):
    cameras = model_info["cameras"]
    spawn_cameras = model_info["spawn_cameras"]
    prior_images = np.stack([_resize_rgb(image_by_name[name], image_by_name[name].shape[1::-1]) for name in spawn_cameras], axis=0)
    prior_camera_t = torch.from_numpy(prior_images.astype(np.float32).transpose(0, 3, 1, 2) / 255.0).unsqueeze(0).to(device)
    lidar_t = torch.from_numpy(lidar_bev.astype(np.float32)).unsqueeze(0).to(device)
    scalar_t = torch.from_numpy(scalar.astype(np.float32)).unsqueeze(0).to(device)
    layout_t = torch.from_numpy(layout.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        inputs = prepare_transfuserpp_inputs(
            scalar=scalar_t,
            camera=prior_camera_t,
            lidar=lidar_t,
            cameras=spawn_cameras,
            config=model_info["prior_config"],
            command_mode=model_info["command_mode"],
            tfpp_camera=model_info["tfpp_camera"],
        )
        outputs = model_info["prior_net"](**inputs)
        pred_target_speed = outputs[1]
        pred_checkpoint = outputs[2]
        base_target = base_target_from_checkpoint(
            pred_checkpoint=pred_checkpoint,
            pred_target_speed=pred_target_speed,
            scalar=scalar_t,
            config=model_info["prior_config"],
            target_dim=model_info["target_dim"],
            speed_dim=model_info["speed_dim"],
        )
        if pred_checkpoint is None:
            checkpoint_flat = torch.zeros((1, model_info["checkpoint_dim"]), dtype=scalar_t.dtype, device=device)
        else:
            checkpoint_flat = _flatten_checkpoint(pred_checkpoint, model_info["checkpoint_dim"])
        if pred_target_speed is None:
            speed_logits = torch.zeros((1, model_info["speed_classes"]), dtype=scalar_t.dtype, device=device)
        else:
            speed_logits = pred_target_speed[:, : model_info["speed_classes"]]
            if speed_logits.shape[1] < model_info["speed_classes"]:
                pad = torch.zeros((1, model_info["speed_classes"] - speed_logits.shape[1]), dtype=speed_logits.dtype, device=device)
                speed_logits = torch.cat([speed_logits, pad], dim=1)
        expected_speed = speed_expectation(pred_target_speed, model_info["prior_config"], 1, device)

        if model_info["is_visual"]:
            visual_images = np.stack([_resize_rgb(image_by_name[name], model_info["visual_size"]) for name in cameras], axis=0)
            visual_camera_t = torch.from_numpy(visual_images.astype(np.float32).transpose(0, 3, 1, 2) / 255.0).unsqueeze(0).to(device)
            visual_lidar = _resize_lidar(lidar_bev, model_info["visual_lidar_size"])
            visual_lidar_t = torch.from_numpy(visual_lidar.astype(np.float32)).unsqueeze(0).to(device)
            visual_lidar_t = _match_lidar_channels(visual_lidar_t, int(model_info["adapter"].visual.lidar_encoder.net[0].in_channels))
            out = model_info["adapter"](
                scalar_t,
                layout_t,
                base_target,
                checkpoint_flat,
                speed_logits,
                expected_speed,
                visual_camera_t,
                visual_lidar_t,
            )
        else:
            out = model_info["adapter"](scalar_t, layout_t, base_target, checkpoint_flat, speed_logits, expected_speed)

    pred = out["target"].detach().cpu()[0]
    speed_dim = int(model_info["speed_dim"])
    traj_dim = int(pred.shape[0]) - speed_dim - 1
    traj = pred[:traj_dim].reshape(-1, 3).numpy().astype(np.float32)
    speeds = pred[traj_dim : traj_dim + speed_dim].numpy().astype(np.float32)
    stop_prob = float(torch.sigmoid(pred[-1]).item())
    stop_state = int(torch.argmax(out["stop_state"].detach().cpu()[0]).item())
    stop_reason = int(torch.argmax(out["stop_reason"].detach().cpu()[0]).item())
    return traj, speeds, stop_prob, stop_state, stop_reason


def _red_light_infraction(carla, vehicle, odom_speed, seen_red_lights, infractions, location, args):
    if not args.count_red_lights or odom_speed <= args.red_light_speed_threshold:
        return
    traffic_light = vehicle.get_traffic_light()
    if traffic_light is None:
        return
    if vehicle.get_traffic_light_state() == carla.TrafficLightState.Red and traffic_light.id not in seen_red_lights:
        seen_red_lights.add(traffic_light.id)
        infractions["red_light"].append(f"Agent moved through red light {traffic_light.id} at {_location_text(location)}")


def rollout(args):
    carla = _import_carla()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_info = _load_cached_adapter(args, device)
    route, meta, episode_dir = _load_token_route(args.route_source, args.episode_index)
    route_len = float(route[-1, 3])
    layout = _episode_layout_vector(Path(episode_dir), model_info["layout_dim"])

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    map_name = args.map or meta.get("map", "")
    if map_name:
        requested_map = _short_map_name(map_name)
        current_map = _short_map_name(world.get_map().name)
        if requested_map and requested_map != current_map:
            world = client.load_world(requested_map)

    original_settings = world.get_settings()
    actors = []
    video_writer = None
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.hz
        settings.no_rendering_mode = args.no_rendering
        world.apply_settings(settings)

        blueprints = world.get_blueprint_library()
        vehicle_bp = blueprints.filter(args.vehicle_filter)[0]
        start = route[min(args.start_index, len(route) - 1)]
        spawn = carla.Transform(
            carla.Location(x=float(start[0]), y=float(start[1]), z=args.spawn_z),
            carla.Rotation(yaw=math.degrees(float(start[2]))),
        )
        vehicle = world.spawn_actor(vehicle_bp, spawn)
        actors.append(vehicle)
        vehicle.apply_control(carla.VehicleControl(brake=1.0))

        camera_queues = _spawn_cameras(carla, world, blueprints, vehicle, model_info["spawn_cameras"], args, actors)
        lidar_q = _spawn_lidar(carla, world, blueprints, vehicle, args, actors)
        imu_q = _spawn_imu(carla, world, blueprints, vehicle, args, actors)

        video_camera_q = None
        video_size = args.video_image_size or args.image_size
        if args.video_output and args.video_image_size:
            video_bp = _camera_blueprint(blueprints, video_size, args.camera_fov, args.hz)
            video_camera = world.spawn_actor(video_bp, _camera_transform(carla, "front"), attach_to=vehicle)
            actors.append(video_camera)
            video_camera_q = queue.Queue()
            video_camera.listen(video_camera_q.put)
        video_writer = _open_video_writer(args.video_output, video_size, args.hz, args.video_scale, args.video_codec)

        infractions = _new_infraction_log()
        seen_red_lights = set()
        collision_bp = blueprints.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        actors.append(collision_sensor)

        def on_collision(event):
            other_type = event.other_actor.type_id if event.other_actor else "unknown"
            loc = event.transform.location
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x * impulse.x + impulse.y * impulse.y + impulse.z * impulse.z)
            infractions[_collision_key(other_type)].append(f"Collision with {other_type} at {_location_text(loc)} intensity={intensity:.2f}")

        collision_sensor.listen(on_collision)

        for _ in range(max(int(args.warmup_sec * args.hz), 0)):
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            world.tick()

        max_steps = int(args.duration_sec * args.hz)
        cross_track_errors = []
        progress_values = []
        success = False
        route_deviation = False
        previous_route_idx = args.start_index
        control_state = {}
        stop_probs = []
        stop_states = []
        stop_reasons = []
        for step in range(max_steps):
            frame = world.tick()
            image_by_name = {
                name: _carla_image_to_rgb(_get_matching(sensor_q, frame), args.image_size)
                for name, sensor_q in camera_queues.items()
            }
            lidar_bev = _carla_lidar_to_bev(_get_matching(lidar_q, frame), args).astype(np.float32)
            imu_data = _get_matching(imu_q, frame)
            video_image = (
                _carla_image_to_rgb(_get_matching(video_camera_q, frame), video_size)
                if video_camera_q is not None
                else image_by_name["front"]
            )

            transform = vehicle.get_transform()
            location = transform.location
            yaw = math.radians(transform.rotation.yaw)
            angular_velocity = vehicle.get_angular_velocity()
            odom = np.asarray([location.x, location.y, yaw, _projected_speed(vehicle), math.radians(float(angular_velocity.z))], dtype=np.float32)
            imu_values = np.asarray(
                [
                    imu_data.accelerometer.x,
                    imu_data.accelerometer.y,
                    imu_data.accelerometer.z,
                    imu_data.gyroscope.x,
                    imu_data.gyroscope.y,
                    imu_data.gyroscope.z,
                ],
                dtype=np.float32,
            )
            scalar, nearest_idx, route_dist = _make_scalar_monotonic(route, route_len, odom, imu_values, True, True, previous_route_idx, args)
            previous_route_idx = nearest_idx
            traj, speeds, stop_prob, stop_state, stop_reason = _predict(model_info, device, scalar, layout, image_by_name, lidar_bev)
            _apply_control(carla, vehicle, traj, speeds, stop_prob, args, control_state)

            cross_track_errors.append(route_dist)
            progress_m = float(route[nearest_idx, 3])
            progress_values.append(progress_m)
            stop_probs.append(stop_prob)
            stop_states.append(stop_state)
            stop_reasons.append(stop_reason)
            _red_light_infraction(carla, vehicle, float(odom[3]), seen_red_lights, infractions, location, args)
            route_completion_pct = 100.0 * min(max(progress_m / max(route_len, 1e-6), 0.0), 1.0)
            scores_now = _score_like_leaderboard(route_completion_pct, infractions)
            if video_writer is not None:
                frame_bgr = _render_video_frame(video_image, route, odom, traj, progress_m, route_len, route_dist, scores_now, step, args)
                line = (
                    f"{model_info['mode']}  stop {stop_prob:.2f} state {stop_state} reason {stop_reason} "
                    f"cmd_v {control_state.get('desired_speed', 0.0):.2f}"
                )
                cv2.putText(frame_bgr, line, (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame_bgr, line, (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                video_writer.write(frame_bgr)

            if step == 0 or (step + 1) % max(int(args.report_every_sec * args.hz), 1) == 0:
                print(
                    f"step={step + 1}/{max_steps} route={route_completion_pct:.1f}% "
                    f"cte={route_dist:.2f}m speed={odom[3]:.2f} cmd_v={control_state.get('desired_speed', 0.0):.2f} "
                    f"stop={stop_prob:.2f}",
                    flush=True,
                )
            if route_len - route[nearest_idx, 3] <= args.goal_tolerance_m:
                success = True
                break
            if route_dist > args.failure_distance_m:
                route_deviation = True
                infractions["route_dev"].append(f"Agent deviated from route at {_location_text(location)} distance={route_dist:.2f}m")
                break

        if not success and not route_deviation and len(cross_track_errors) >= max_steps:
            infractions["route_timeout"].append(f"Route timeout after {args.duration_sec:.1f}s")
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        final_progress = progress_values[-1] if progress_values else 0.0
        max_progress = max(progress_values) if progress_values else 0.0
        route_completion_pct = 100.0 * min(max(max_progress / max(route_len, 1e-6), 0.0), 1.0)
        scores = _score_like_leaderboard(route_completion_pct, infractions)
        metrics = {
            "route_source": str(episode_dir),
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "model_mode": model_info["mode"],
            "status": "Completed" if success else ("Failed - Agent deviated from the route" if route_deviation else "Failed - Route timeout"),
            "success": success,
            "steps": len(cross_track_errors),
            "infractions": infractions,
            "scores": scores,
            "route_length_m": route_len,
            "final_progress_m": final_progress,
            "max_progress_m": max_progress,
            "route_completion_pct": route_completion_pct,
            "mean_cross_track_error_m": float(np.mean(cross_track_errors)) if cross_track_errors else None,
            "max_cross_track_error_m": float(np.max(cross_track_errors)) if cross_track_errors else None,
            "mean_stop_prob": float(np.mean(stop_probs)) if stop_probs else None,
            "max_stop_prob": float(np.max(stop_probs)) if stop_probs else None,
            "stop_state_counts": {str(k): int(v) for k, v in zip(*np.unique(np.asarray(stop_states, dtype=np.int64), return_counts=True))} if stop_states else {},
            "stop_reason_counts": {str(k): int(v) for k, v in zip(*np.unique(np.asarray(stop_reasons, dtype=np.int64), return_counts=True))} if stop_reasons else {},
            "device": str(device),
            "video_output": args.video_output or None,
            "transfuserpp_load_info": model_info["prior_load_info"],
        }
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    finally:
        if video_writer is not None:
            video_writer.release()
        _destroy_actors(client, carla, actors)
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Closed-loop CARLA rollout for cached TransFuser++ Teach2Drive adapters.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--map", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--route-source", required=True, help="Token episode directory or token_index.npz.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--garage-root", default="")
    parser.add_argument("--team-config", default="")
    parser.add_argument("--tfpp-checkpoint", default="")
    parser.add_argument("--cameras", default="")
    parser.add_argument("--tfpp-camera", default="")
    parser.add_argument("--command-mode", choices=["", "lane_follow", "target_angle"], default="")
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--warmup-sec", type=float, default=1.0)
    parser.add_argument("--hz", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=30)
    parser.add_argument("--spawn-z", type=float, default=0.6)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--x-min", type=float, default=-8.0)
    parser.add_argument("--x-max", type=float, default=20.0)
    parser.add_argument("--y-min", type=float, default=-14.0)
    parser.add_argument("--y-max", type=float, default=14.0)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--lidar-z", type=float, default=1.8)
    parser.add_argument("--lidar-channels", type=int, default=32)
    parser.add_argument("--lidar-range", type=float, default=60.0)
    parser.add_argument("--lidar-points-per-second", type=int, default=180000)
    parser.add_argument("--lidar-upper-fov", type=float, default=10.0)
    parser.add_argument("--lidar-lower-fov", type=float, default=-25.0)
    parser.add_argument("--lookahead-m", type=float, default=8.0)
    parser.add_argument("--heading-score-weight", type=float, default=0.2)
    parser.add_argument("--control-point-index", type=int, default=1)
    parser.add_argument("--control-horizon-sec", type=float, default=1.0)
    parser.add_argument("--wheelbase-m", type=float, default=2.8)
    parser.add_argument("--max-steer-rad", type=float, default=0.6)
    parser.add_argument("--min-speed", type=float, default=0.4)
    parser.add_argument("--max-speed", type=float, default=5.0)
    parser.add_argument("--speed-head-mix", type=float, default=0.0)
    parser.add_argument("--speed-kp", type=float, default=0.35)
    parser.add_argument("--brake-kp", type=float, default=0.5)
    parser.add_argument("--max-throttle", type=float, default=0.45)
    parser.add_argument("--max-brake", type=float, default=0.7)
    parser.add_argument("--max-accel-mps2", type=float, default=1.5)
    parser.add_argument("--max-decel-mps2", type=float, default=2.5)
    parser.add_argument("--steer-smoothing", type=float, default=0.65)
    parser.add_argument("--use-stop-head", action="store_true")
    parser.add_argument("--stop-prob-threshold", type=float, default=0.85)
    parser.add_argument("--goal-tolerance-m", type=float, default=5.0)
    parser.add_argument("--failure-distance-m", type=float, default=12.0)
    parser.add_argument("--route-search-back", type=int, default=15)
    parser.add_argument("--route-search-ahead", type=int, default=300)
    parser.add_argument("--count-red-lights", action="store_true")
    parser.add_argument("--red-light-speed-threshold", type=float, default=0.3)
    parser.add_argument("--report-every-sec", type=float, default=5.0)
    parser.add_argument("--video-output", default="")
    parser.add_argument("--video-image-size", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--video-scale", type=float, default=1.0)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layout-hidden-dim", type=int, default=128)
    parser.add_argument("--visual-dim", type=int, default=256)
    parser.add_argument("--visual-token-dim", type=int, default=192)
    parser.add_argument("--visual-layers", type=int, default=2)
    parser.add_argument("--visual-heads", type=int, default=4)
    parser.add_argument("--visual-image-size", type=int, nargs=2, default=[320, 180], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--visual-lidar-size", type=int, default=128)
    parser.add_argument("--visual-lidar-channels", type=int, default=3)
    parser.add_argument("--speed-dim", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main():
    rollout(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
