#!/usr/bin/env python3
"""Generate lidar-projected sparse depth GT sidecars for v12 training.

For each t2d episode, re-reads the raw carla_garage .laz point clouds, projects
them into the SHIFTED front camera, crops to the model's input region (top 384
rows of the 1024x512 frame), downsamples to a compact grid, and encodes depth
with CARLA's public normalization (clip(m,0,50)/50). 0 == invalid (no lidar hit).

This is NON-PRIVILEGED: the target comes from the lidar sensor projected into the
camera, using only the known extrinsics + CARLA's fixed depth encoding formula.
No depth camera is used.

Output: <episode>/depth_gt/<step:06d>.npy  (float16 [GH,GW], 0=invalid)

Mapping: t2d frame 'step' -> raw stem = idxs[step] where idxs = sorted measurement
indices of the source route (matches convert_pdm_lite_to_t2d enumeration).
"""
import argparse, gzip, json, os
from pathlib import Path
import numpy as np
import laspy

# ---- shifted front camera extrinsics (ego/ground frame, z-up, ground=0) ----
CAM_X, CAM_Y, CAM_Z = 1.25, 0.0, 1.95
W, H, FOV = 1024, 512, 110.0
FOCAL = (W / 2.0) / np.tan(np.deg2rad(FOV / 2.0))
CX, CY = W / 2.0, H / 2.0
CROP_H = 384                       # model crops top 384 rows of the 512-tall frame
Y_SIGN = 1                         # validated: +1 (spearman 0.986 vs saved depth)
DEPTH_CLIP_M = 50.0                # CARLA: clip(m,0,50)/50 -> [0,1]
# storage grid (1/4 of 384x1024)
GH, GW = 96, 256


def _frame_indices(route: Path):
    mdir = route / "measurements"
    return sorted(int(f.name.split(".")[0]) for f in mdir.glob("*.json.gz"))


def _read_laz_xyz(p: Path) -> np.ndarray:
    las = laspy.read(str(p))
    return np.stack([np.asarray(las.x, np.float32),
                     np.asarray(las.y, np.float32),
                     np.asarray(las.z, np.float32)], axis=1)


def project_depth_gt(pts: np.ndarray) -> np.ndarray:
    """pts (N,3) ground frame -> (GH,GW) float16 encoded depth, 0=invalid."""
    X = pts[:, 0] - CAM_X
    Y = Y_SIGN * pts[:, 1] - CAM_Y
    Z = pts[:, 2] - CAM_Z
    m = X > 0.5
    X, Y, Z = X[m], Y[m], Z[m]
    u = CX + FOCAL * (Y / X)
    v = CY - FOCAL * (Z / X)
    depth = X
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < CROP_H)  # crop to top 384 rows
    u, v, depth = u[inb], v[inb], depth[inb]
    # downsample coords to storage grid
    gx = np.clip((u * (GW / W)).astype(np.int32), 0, GW - 1)
    gy = np.clip((v * (GH / CROP_H)).astype(np.int32), 0, GH - 1)
    enc = np.clip(depth / DEPTH_CLIP_M, 0.0, 1.0).astype(np.float32)
    # nearest-surface: keep the SMALLEST depth per cell (front-most occluder)
    out = np.full((GH, GW), np.inf, dtype=np.float32)
    flat = gy * GW + gx
    np.minimum.at(out.reshape(-1), flat, enc)
    out[~np.isfinite(out)] = 0.0   # 0 == invalid
    return out.astype(np.float16)


def process_episode(ep_dir: Path, src_root: Path, overwrite: bool) -> tuple[int, int]:
    meta_p = ep_dir / "episode_meta.json"
    if not meta_p.exists():
        print(f"[skip] {ep_dir.name}: no episode_meta.json", flush=True)
        return 0, 0
    meta = json.loads(meta_p.read_text())
    route = src_root / meta.get("source_route", "")
    if not route.is_dir():
        print(f"[skip] {ep_dir.name}: raw route missing {route.name}", flush=True)
        return 0, 0
    idxs = _frame_indices(route)
    out_dir = ep_dir / "depth_gt"
    out_dir.mkdir(exist_ok=True)
    n_ok = n_skip = 0
    with (ep_dir / "frames.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            step = int(rec["step"])
            outp = out_dir / f"{step:06d}.npy"
            if outp.exists() and not overwrite:
                n_skip += 1
                continue
            if step >= len(idxs):
                continue
            stem = f"{idxs[step]:04d}"
            laz = route / "lidar" / f"{stem}.laz"
            if not laz.exists():
                continue
            try:
                gt = project_depth_gt(_read_laz_xyz(laz))
            except Exception as e:
                print(f"[warn] {ep_dir.name} step {step}: {e}", flush=True)
                continue
            np.save(outp, gt)
            n_ok += 1
    return n_ok, n_skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-root", default=os.path.expanduser(
        "~/dataset/byeongjae/datasets/t2d_local_full"))
    ap.add_argument("--src-root", default=os.path.expanduser(
        "~/dataset/byeongjae/datasets/pdm_lite_tesla_front_triplet_shifted_full/data"))
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ep_root = Path(args.episodes_root)
    src_root = Path(args.src_root)
    eps = sorted(p for p in ep_root.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    eps = eps[args.shard::args.num_shards]
    print(f"[gen_depth_gt] shard {args.shard}/{args.num_shards}: {len(eps)} episodes "
          f"grid={GH}x{GW} focal={FOCAL:.1f}", flush=True)
    tot_ok = tot_skip = 0
    for k, ep in enumerate(eps):
        ok, sk = process_episode(ep, src_root, args.overwrite)
        tot_ok += ok; tot_skip += sk
        print(f"[{k+1}/{len(eps)}] {ep.name} ok={ok} skip={sk} cum_ok={tot_ok}", flush=True)
    print(f"[done] shard {args.shard}: frames written={tot_ok} skipped={tot_skip}", flush=True)


if __name__ == "__main__":
    main()
