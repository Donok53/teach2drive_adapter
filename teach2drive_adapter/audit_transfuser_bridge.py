import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .transfuser_bridge import (
    TransFuserInputSpec,
    crop_rgb_like_transfuser,
    lidar_bev_to_transfuser,
    load_frame_record,
    stitch_camera_views,
    target_point_from_scalar,
)


def _load_index(index_path: Path):
    return np.load(index_path, allow_pickle=True)


def _resolve_episode_dir(raw_dir: str, override_root: str) -> Path:
    path = Path(str(raw_dir))
    if not override_root:
        return path
    return Path(override_root).expanduser().resolve() / path.name


def _save_preview(output: Path, stitched: np.ndarray, cropped_chw: np.ndarray, lidar_tf: np.ndarray) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cropped = np.transpose(cropped_chw, (1, 2, 0)).astype(np.uint8)
    lidar_vis = np.concatenate(lidar_tf, axis=1)
    lidar_vis = (np.clip(lidar_vis, 0.0, 1.0) * 255).astype(np.uint8)
    lidar_vis = cv2.cvtColor(lidar_vis, cv2.COLOR_GRAY2RGB)
    scale_w = cropped.shape[1]
    stitched_small = cv2.resize(stitched, (scale_w, int(stitched.shape[0] * scale_w / stitched.shape[1])), interpolation=cv2.INTER_AREA)
    lidar_small = cv2.resize(lidar_vis, (scale_w, int(lidar_vis.shape[0] * scale_w / lidar_vis.shape[1])), interpolation=cv2.INTER_NEAREST)
    preview = np.concatenate([stitched_small, cropped, lidar_small], axis=0)
    cv2.imwrite(str(output), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))


def audit(args: argparse.Namespace) -> None:
    spec = TransFuserInputSpec()
    index = _load_index(Path(args.index).expanduser())
    sample_idx = int(args.sample_index)
    episode_idx = int(index["sample_episode_indices"][sample_idx])
    frame_idx = int(index["sample_frame_indices"][sample_idx])
    episode_dir = _resolve_episode_dir(index["episode_dirs"][episode_idx], args.episode_root_override)
    frame = load_frame_record(episode_dir, frame_idx)
    available_cameras = {item.strip() for item in args.available_cameras.split(",") if item.strip()}
    if available_cameras:
        frame = dict(frame)
        frame["camera_tokens"] = {
            camera: token
            for camera, token in frame.get("camera_tokens", {}).items()
            if camera in available_cameras
        }

    camera_order = [item.strip() for item in args.camera_order.split(",") if item.strip()]
    stitched, camera_present = stitch_camera_views(
        episode_dir,
        frame,
        camera_order=camera_order,
        missing_policy=args.missing_camera_policy,
    )
    cropped = crop_rgb_like_transfuser(stitched, crop_hw=spec.image_hw)

    lidar_token = frame.get("lidar_bev_token")
    raw_lidar_shape = None
    if lidar_token:
        lidar_path = episode_dir / lidar_token
        lidar_bev = np.load(lidar_path).astype(np.float32)
        raw_lidar_shape = list(lidar_bev.shape)
        lidar_tf = lidar_bev_to_transfuser(lidar_bev, spec)
    else:
        lidar_path = None
        lidar_tf = np.zeros((spec.lidar_channels, *spec.lidar_hw), dtype=np.float32)

    scalar = index["scalar_features"][sample_idx].astype(np.float32)
    target_point = target_point_from_scalar(scalar)

    dataset_meta_path = episode_dir.parent / "dataset_meta.json"
    dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8")) if dataset_meta_path.exists() else {}

    report = {
        "index": str(Path(args.index).expanduser()),
        "sample_index": sample_idx,
        "episode_index": episode_idx,
        "frame_index": frame_idx,
        "episode_dir": str(episode_dir),
        "dataset_meta": {
            "image_size_wh": dataset_meta.get("image_size_wh"),
            "camera_fov": dataset_meta.get("camera_fov", "unknown"),
            "hz": dataset_meta.get("hz"),
            "cameras": dataset_meta.get("cameras"),
            "lidar_bev_size": dataset_meta.get("lidar_bev_size"),
        },
        "transfuser_expected": {
            "camera_order": list(spec.expected_camera_order),
            "image_chw": [3, spec.image_hw[0], spec.image_hw[1]],
            "camera_fov_deg": spec.expected_camera_fov_deg,
            "lidar_chw": [spec.lidar_channels, spec.lidar_hw[0], spec.lidar_hw[1]],
            "lidar_source": "raw point cloud histogram in official TransFuser",
        },
        "bridge_output": {
            "requested_camera_order": camera_order,
            "available_cameras_override": sorted(available_cameras) if available_cameras else None,
            "missing_camera_policy": args.missing_camera_policy,
            "camera_present": camera_present,
            "stitched_hwc": list(stitched.shape),
            "rgb_chw": list(cropped.shape),
            "raw_lidar_shape": raw_lidar_shape,
            "lidar_chw": list(lidar_tf.shape),
            "lidar_path": str(lidar_path) if lidar_path else None,
            "velocity_mps": float(scalar[0]),
            "target_point_xy": target_point.astype(float).tolist(),
        },
        "compatibility_notes": [
            "Camera tensors can be formed even with missing side cameras by repeat/zero policy.",
            "Current Teach2Drive LiDAR BEV is an approximation, not the raw point-cloud histogram used by original TransFuser.",
            "The current CARLA dataset was collected with 90 degree camera FOV, while TransFuser expects 120 degree cameras; this is a domain gap for pretrained weights.",
            "Closed-loop destination driving still needs a route/goal token; this audit uses the lookahead scalar as local target_point.",
        ],
    }

    if args.preview:
        _save_preview(Path(args.preview).expanduser(), stitched, cropped, lidar_tf)
        report["preview"] = str(Path(args.preview).expanduser())

    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Teach2Drive samples against TransFuser input assumptions.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--episode-root-override", default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--camera-order", default="left,front,right")
    parser.add_argument("--available-cameras", default="", help="Optional comma list to simulate a robot with only these cameras.")
    parser.add_argument("--missing-camera-policy", choices=["repeat_front", "zero"], default="repeat_front")
    parser.add_argument("--output", default="")
    parser.add_argument("--preview", default="")
    return parser


def main() -> None:
    audit(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
