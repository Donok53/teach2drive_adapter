import argparse
import json
import math
import queue
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2] / "teach2drive_bootstrap"
if BOOTSTRAP_ROOT.exists() and str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

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

from .transfuser_adapter_model import TransFuserResidualAdapterPolicy


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
    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=1.8)), attach_to=vehicle)
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


def _load_adapter_model(args, device):
    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    cameras = list(checkpoint.get("cameras") or ckpt_args.get("cameras", "left,front,right").split(","))
    cameras = [item.strip() for item in cameras if str(item).strip()]
    model = TransFuserResidualAdapterPolicy(
        transfuser_root=args.transfuser_root or ckpt_args.get("transfuser_root", "/home/byeongjae/code/transfuser"),
        team_config=args.team_config or ckpt_args.get("team_config", "/home/byeongjae/code/transfuser/model_ckpt/models_2022/transfuser"),
        device=device,
        scalar_dim=int(checkpoint.get("scalar_dim", 20)),
        target_dim=int(checkpoint.get("target_dim", 17)),
        hidden_dim=int(ckpt_args.get("hidden_dim", args.hidden_dim)),
        speed_dim=int(ckpt_args.get("speed_dim", 4)),
    )
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    print(f"loaded_adapter={args.checkpoint} missing={len(missing)} unexpected={len(unexpected)} cameras={cameras}", flush=True)
    model.to(device)
    model.eval()
    return model, cameras


def _predict_adapter(model, device, scalar, images, lidar):
    scalar_t = torch.from_numpy(scalar.astype(np.float32)).unsqueeze(0).to(device)
    camera_t = torch.from_numpy(images.astype(np.float32).transpose(0, 3, 1, 2) / 255.0).unsqueeze(0).to(device)
    lidar_t = torch.from_numpy(lidar.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(scalar_t, camera_t, lidar_t)["target"].detach().cpu().numpy()[0]
    traj = out[: model.traj_dim].reshape(-1, 3)
    speeds = out[model.traj_dim : model.traj_dim + model.speed_dim]
    stop_prob = float(1.0 / (1.0 + np.exp(-float(out[-1]))))
    return traj.astype(np.float32), speeds.astype(np.float32), stop_prob


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
    model, cameras = _load_adapter_model(args, device)
    route, meta, episode_dir = _load_token_route(args.route_source, args.episode_index)
    route_len = float(route[-1, 3])

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
        start = route[args.start_index]
        spawn = carla.Transform(
            carla.Location(x=float(start[0]), y=float(start[1]), z=args.spawn_z),
            carla.Rotation(yaw=math.degrees(float(start[2]))),
        )
        vehicle = world.spawn_actor(vehicle_bp, spawn)
        actors.append(vehicle)
        vehicle.apply_control(carla.VehicleControl(brake=1.0))

        camera_queues = _spawn_cameras(carla, world, blueprints, vehicle, cameras, args, actors)
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
        for step in range(max_steps):
            frame = world.tick()
            image_by_name = {
                name: _carla_image_to_rgb(_get_matching(sensor_q, frame), args.image_size)
                for name, sensor_q in camera_queues.items()
            }
            images = np.stack([image_by_name[name] for name in cameras], axis=0)
            lidar_bev = _carla_lidar_to_bev(_get_matching(lidar_q, frame), args)
            imu_data = _get_matching(imu_q, frame)
            video_image = (
                _carla_image_to_rgb(_get_matching(video_camera_q, frame), video_size)
                if video_camera_q is not None
                else image_by_name["front" if "front" in image_by_name else cameras[0]]
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
            traj, speeds, stop_prob = _predict_adapter(model, device, scalar, images, lidar_bev)
            _apply_control(carla, vehicle, traj, speeds, stop_prob, args, control_state)

            cross_track_errors.append(route_dist)
            progress_m = float(route[nearest_idx, 3])
            progress_values.append(progress_m)
            _red_light_infraction(carla, vehicle, float(odom[3]), seen_red_lights, infractions, location, args)
            route_completion_pct = 100.0 * min(max(progress_m / max(route_len, 1e-6), 0.0), 1.0)
            scores_now = _score_like_leaderboard(route_completion_pct, infractions)
            if video_writer is not None:
                frame_bgr = _render_video_frame(video_image, route, odom, traj, progress_m, route_len, route_dist, scores_now, step, args)
                line = f"adapter closed-loop  stop {stop_prob:.2f}  v_pred {speeds[min(args.control_point_index, len(speeds)-1)]:.2f}"
                cv2.putText(frame_bgr, line, (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame_bgr, line, (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                video_writer.write(frame_bgr)

            if step == 0 or (step + 1) % max(int(args.report_every_sec * args.hz), 1) == 0:
                print(
                    f"step={step + 1}/{max_steps} route={route_completion_pct:.1f}% "
                    f"cte={route_dist:.2f}m speed={odom[3]:.2f} cmd_v={control_state.get('desired_speed', 0.0):.2f} stop={stop_prob:.2f}",
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
            "device": str(device),
            "video_output": args.video_output or None,
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
    parser = argparse.ArgumentParser(description="Closed-loop CARLA rollout for a frozen TransFuser + Teach2Drive residual adapter.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--map", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--route-source", required=True, help="Token episode directory or token_index.npz.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transfuser-root", default="")
    parser.add_argument("--team-config", default="")
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
    parser.add_argument("--stop-prob-threshold", type=float, default=0.95)
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
    parser.add_argument("--cpu", action="store_true")
    return parser


def main():
    rollout(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
