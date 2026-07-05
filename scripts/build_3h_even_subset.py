#!/usr/bin/env python3
"""Build an even ~3h subset by COPYING completed route dirs from the live
pdm_lite collection into a new, separate folder. Does not touch the source."""
import os, random, shutil, json, time, sys

SRC = "/home/jovyan/dataset/byeongjae/datasets/pdm_lite_tesla_front_triplet_shifted_full/data"
DST_ROOT = "/home/jovyan/dataset/byeongjae/datasets/pdm_lite_front_triplet_shifted_3h_subset"
DST = os.path.join(DST_ROOT, "data")
SAVE_HZ = 4.0                       # data_save_freq=5, carla_fps=20 -> 4 Hz
TARGET_SECONDS = 3 * 3600           # 3 hours of driving
TARGET_FRAMES = int(TARGET_SECONDS * SAVE_HZ)   # 43200
SETTLE_SEC = 300                    # measurements/ untouched >5 min => route finished
MIN_FRAMES = 5                      # drop degenerate/aborted stubs
SEED = 42

os.makedirs(DST, exist_ok=True)
now = time.time()
log = lambda *a: print(*a, flush=True)

log(f"[scan] listing candidates under {SRC}")
names = [n for n in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, n))]
log(f"[scan] {len(names)} route dirs total")

# Keep only finished, frame-aligned dirs (measurements settled >5min, rgb_front==measurements)
candidates = []
for n in names:
    p = os.path.join(SRC, n)
    md = os.path.join(p, "measurements")
    rf = os.path.join(p, "rgb_front")
    if not (os.path.isdir(md) and os.path.isdir(rf)):
        continue
    try:
        if os.stat(md).st_mtime > now - SETTLE_SEC:
            continue  # still being written
        mfiles = [f for f in os.listdir(md) if f.endswith(".json.gz")]
        nm = len(mfiles)
        nr = len(os.listdir(rf))
    except OSError:
        continue
    if nm < MIN_FRAMES or nm != nr:   # too short or misaligned
        continue
    candidates.append((n, nm))

log(f"[scan] {len(candidates)} finished+aligned candidates")
avail = sum(c[1] for c in candidates)
log(f"[scan] available: {avail} frames = {avail/SAVE_HZ/3600:.2f} h")

# Seeded shuffle -> representative mix across the round-robin town order
random.seed(SEED)
random.shuffle(candidates)

selected, total = [], 0
for n, nm in candidates:
    selected.append((n, nm)); total += nm
    if total >= TARGET_FRAMES:
        break

log(f"[plan] selected {len(selected)} routes = {total} frames = {total/SAVE_HZ/3600:.3f} h")
if total < TARGET_FRAMES:
    log(f"[warn] only {total/SAVE_HZ/3600:.2f}h available (< 3h target); copying all of it")

# Copy
copied_frames = 0
for i, (n, nm) in enumerate(selected, 1):
    s = os.path.join(SRC, n)
    d = os.path.join(DST, n)
    if os.path.exists(d):
        copied_frames += nm
        log(f"[{i}/{len(selected)}] exists, skip {n}")
        continue
    shutil.copytree(s, d)
    copied_frames += nm
    log(f"[{i}/{len(selected)}] copied {n} ({nm}f)  cum={copied_frames}f={copied_frames/SAVE_HZ/3600:.3f}h")

manifest = {
    "source": SRC,
    "save_hz": SAVE_HZ,
    "target_frames": TARGET_FRAMES,
    "selected_routes": len(selected),
    "total_frames": total,
    "sim_seconds": total / SAVE_HZ,
    "hours": total / SAVE_HZ / 3600,
    "seed": SEED,
    "routes": [n for n, _ in selected],
}
with open(os.path.join(DST_ROOT, "subset_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
log(f"[done] {len(selected)} routes, {total/SAVE_HZ/3600:.3f} h -> {DST_ROOT}")
