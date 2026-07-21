#!/usr/bin/env python3
"""
Convert a carla_garage-format pdm_lite subset -> t2d token episodes that
`teach2drive.token_dataset` can index directly (no export_paired_profile_view).

Source route dir (carla_garage):
    <route>/rgb_left|rgb_front|rgb_right/NNNN.jpg
            lidar/NNNN.laz
            measurements/NNNN.json.gz   (pos_global, theta, speed, steer, throttle, brake, ...)

Output episode (t2d, single-profile, token_dataset-ready):
    episode_XXXXXX/
        camera/{left,front,right}/NNNNNN_<tok>.jpg   (symlink to source jpg)
        lidar_bev/NNNNNN_<tok>.npy                   (3xHxW BEV via _points_to_bev)
        frames.jsonl                                 (one record per frame)
        rigs/<profile>/sensor_layout.json            (extrinsics, for EXTRINSIC_AWARE training)
        episode_meta.json, episode_summary.json

FIRST DRAFT — validate against real data:
  * lidar frame / axis convention (see --flip-y, --lidar-to-ego)
  * theta units (assumed radians, carla_garage convention)
"""
import argparse, gzip, json, os, sys, uuid, math
from pathlib import Path
import numpy as np


def _points_to_bev(points, grid_size, x_min, x_max, y_min, y_max, z_min, z_max):
    """Inlined verbatim from teach2drive.carla_collect._points_to_bev so the
    converter has no dependency on the (carla-heavy) bootstrap package import.
    Returns a (3, grid, grid) float16 BEV: [occupancy, height, intensity]."""
    occ = np.zeros((grid_size, grid_size), dtype=np.float32)
    height = np.zeros((grid_size, grid_size), dtype=np.float32)
    intensity = np.zeros((grid_size, grid_size), dtype=np.float32)
    count = np.zeros((grid_size, grid_size), dtype=np.float32)
    if points.size == 0:
        return np.stack([occ, height, intensity], axis=0).astype(np.float16)
    xs = points[:, 0]; ys = points[:, 1]; zs = points[:, 2]
    valid = (np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
             & (xs >= x_min) & (xs < x_max) & (ys >= y_min) & (ys < y_max)
             & (zs >= z_min) & (zs <= z_max))
    if not np.any(valid):
        return np.stack([occ, height, intensity], axis=0).astype(np.float16)
    xs = xs[valid]; ys = ys[valid]; zs = zs[valid]
    inten = points[:, 3][valid] if points.shape[1] > 3 else np.ones_like(xs)
    ix = np.clip(((xs - x_min) * grid_size / max(x_max - x_min, 1e-6)).astype(np.int32), 0, grid_size - 1)
    iy = np.clip(((ys - y_min) * grid_size / max(y_max - y_min, 1e-6)).astype(np.int32), 0, grid_size - 1)
    rows = grid_size - 1 - ix; cols = iy
    flat = rows * grid_size + cols
    np.add.at(occ.reshape(-1), flat, 1.0)
    np.add.at(count.reshape(-1), flat, 1.0)
    np.maximum.at(height.reshape(-1), flat, (zs - z_min) / max(z_max - z_min, 1e-6))
    np.add.at(intensity.reshape(-1), flat, inten)
    occ = np.clip(np.log1p(occ) / np.log(16.0), 0.0, 1.0)
    nonzero = count > 0
    if np.any(nonzero):
        intensity[nonzero] = intensity[nonzero] / count[nonzero]
        hi = np.percentile(intensity[nonzero], 95)
        if hi > 1e-6:
            intensity = np.clip(intensity / hi, 0.0, 1.0)
    return np.stack([occ, height, intensity], axis=0).astype(np.float16)

BEV = dict(grid_size=128, x_min=-8.0, x_max=20.0, y_min=-14.0, y_max=14.0, z_min=-2.0, z_max=4.0)
CAMERAS = ("left", "front", "right")
SAVE_HZ = 4.0  # data_save_freq=5, carla_fps=20


def _read_meas(p: Path) -> dict:
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def _read_laz_points(p: Path, flip_y: bool) -> np.ndarray:
    """Return (N,4) float32 [x,y,z,intensity] from a carla_garage .laz lidar file."""
    import laspy
    las = laspy.read(str(p))
    xyz = np.stack([np.asarray(las.x, np.float32),
                    np.asarray(las.y, np.float32),
                    np.asarray(las.z, np.float32)], axis=1)
    if flip_y:
        xyz[:, 1] = -xyz[:, 1]
    try:
        inten = np.asarray(las.intensity, np.float32)
        inten = inten / max(float(inten.max()), 1.0)
    except Exception:
        inten = np.ones(len(xyz), np.float32)
    return np.concatenate([xyz, inten[:, None]], axis=1).astype(np.float32)


def _frame_indices(route: Path) -> list:
    mdir = route / "measurements"
    idxs = sorted(int(f.name.split(".")[0]) for f in mdir.glob("*.json.gz"))
    return idxs


def _odom_from_meas(m: dict) -> dict:
    pos = m.get("pos_global") or [0.0, 0.0]
    x, y = float(pos[0]), float(pos[1])
    yaw = float(m.get("theta", 0.0))          # carla_garage ego yaw (radians)
    v = float(m.get("speed", 0.0))
    return {"x": x, "y": y, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": yaw,
            "v_forward": v, "velocity": [v * math.cos(yaw), v * math.sin(yaw), 0.0],
            "yaw_rate": 0.0}


def _sensor_layout(route: Path, profile: str) -> dict:
    sp = route / f"{profile}_sensor_profile.json"
    if sp.exists():
        prof = json.loads(sp.read_text())
        return {"profile": profile, "cameras": prof.get("cameras", {}),
                "ego_vehicle_model": prof.get("ego_vehicle_model")}
    return {"profile": profile, "cameras": {}}


def convert_route(route: Path, out_root: Path, ep_idx: int, profile: str,
                  flip_y: bool, link: bool) -> dict:
    ep_token = uuid.uuid4().hex
    ep = out_root / f"episode_{ep_idx:06d}"
    for c in CAMERAS:
        (ep / "camera" / c).mkdir(parents=True, exist_ok=True)
    (ep / "lidar_bev").mkdir(parents=True, exist_ok=True)
    (ep / "rigs" / profile).mkdir(parents=True, exist_ok=True)
    (ep / "rigs" / profile / "sensor_layout.json").write_text(
        json.dumps(_sensor_layout(route, profile), indent=2))

    idxs = _frame_indices(route)
    n_written = 0
    with (ep / "frames.jsonl").open("w", encoding="utf-8") as fout:
        for step, i in enumerate(idxs):
            stem = f"{i:04d}"
            cam_src = {}
            for _c in CAMERAS:
                _p = route / f"rgb_{_c}" / f"{stem}.jpg"
                if not _p.exists():
                    _p = route / "rgb" / f"{stem}.jpg"
                cam_src[_c] = _p
            laz = route / "lidar" / f"{stem}.laz"
            meas = route / "measurements" / f"{stem}.json.gz"
            if not (laz.exists() and meas.exists() and all(p.exists() for p in cam_src.values())):
                continue
            tok = uuid.uuid4().hex
            camera_tokens = {}
            for c in CAMERAS:
                dst = ep / "camera" / c / f"{step:06d}_{tok}.jpg"
                if not dst.exists():
                    if link:
                        os.symlink(cam_src[c].resolve(), dst)
                    else:
                        import shutil; shutil.copy2(cam_src[c], dst)
                camera_tokens[c] = str(dst.relative_to(ep))
            # lidar BEV
            pts = _read_laz_points(laz, flip_y)
            bev = _points_to_bev(pts, BEV["grid_size"], BEV["x_min"], BEV["x_max"],
                                 BEV["y_min"], BEV["y_max"], BEV["z_min"], BEV["z_max"])
            lpath = ep / "lidar_bev" / f"{step:06d}_{tok}.npy"
            np.save(lpath, bev.astype(np.float16))
            m = _read_meas(meas)
            rec = {
                "episode_token": ep_token,
                "frame_token": tok,
                "step": int(step),
                "phase": "drive",
                "time": float(step) / SAVE_HZ,
                "camera_tokens": camera_tokens,
                "lidar_bev_token": str(lpath.relative_to(ep)),
                "control": {"steer": float(m.get("steer", 0.0)),
                            "throttle": float(m.get("throttle", 0.0)),
                            "brake": float(m.get("brake", 0.0))},
                "odom": _odom_from_meas(m),
                "imu": {"accelerometer": [0.0, 0.0, 0.0], "gyroscope": [0.0, 0.0, 0.0]},
                "lane": {"valid": False},
                # carry through useful carla_garage supervision (optional downstream)
                "target_point": m.get("target_point"),
                "target_point_next": m.get("target_point_next"),
                "command": m.get("command"),
                "next_command": m.get("next_command"),
                "target_speed": m.get("target_speed"),
                "hazard": {"light": bool(m.get("light_hazard", False)),
                           "stop_sign": bool(m.get("stop_sign_hazard", False)),
                           "vehicle": bool(m.get("vehicle_hazard", False)),
                           "walker": bool(m.get("walker_hazard", False)),
                           "junction": bool(m.get("junction", False))},
                "source_route": route.name,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    (ep / "episode_meta.json").write_text(json.dumps(
        {"episode_token": ep_token, "profile": profile, "source_route": route.name,
         "cameras": list(CAMERAS), "save_hz": SAVE_HZ, "bev": BEV}, indent=2))
    (ep / "episode_summary.json").write_text(json.dumps(
        {"episode_token": ep_token, "frames": n_written,
         "sim_seconds": n_written / SAVE_HZ}, indent=2))
    return {"episode": ep.name, "frames": n_written}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default=os.path.expanduser(
        "~/dataset/byeongjae/datasets/pdm_lite_front_triplet_shifted_3h_subset/data"))
    ap.add_argument("--out-root", default=os.path.expanduser(
        "~/dataset/byeongjae/datasets/t2d_pdm_lite_front_triplet_shifted_3h"))
    ap.add_argument("--profile", default="front_triplet_shifted")
    ap.add_argument("--flip-y", action="store_true",
                    help="negate lidar y (validate against a real BEV first)")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlink")
    ap.add_argument("--limit", type=int, default=0, help="convert only N routes (debug)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    if out_root.exists() and args.overwrite:
        import shutil; shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    routes = sorted(p for p in src_root.iterdir() if p.is_dir())
    if args.limit:
        routes = routes[:args.limit]
    print(f"[convert] {len(routes)} routes -> {out_root}", flush=True)

    summaries, total = [], 0
    for k, route in enumerate(routes):
        try:
            s = convert_route(route, out_root, k, args.profile, args.flip_y, not args.copy)
        except Exception as e:
            print(f"[skip] {route.name}: {e}", flush=True)
            continue
        summaries.append(s); total += s["frames"]
        print(f"[{k+1}/{len(routes)}] {s['episode']} frames={s['frames']} "
              f"cum={total} ({total/SAVE_HZ/3600:.3f}h)", flush=True)

    (out_root / "dataset_summary.json").write_text(json.dumps(
        {"episodes": len(summaries), "frames": total, "hours": total / SAVE_HZ / 3600,
         "profile": args.profile, "src": str(src_root)}, indent=2))
    print(f"[done] {len(summaries)} episodes, {total} frames = {total/SAVE_HZ/3600:.3f}h", flush=True)


if __name__ == "__main__":
    main()
