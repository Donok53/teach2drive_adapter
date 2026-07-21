"""AdaBN: recompute BatchNorm running statistics on target-domain data.

Loads each base TransFuser++ checkpoint, resets BN running stats, forwards
target-domain frames (no gradients, no labels) with BN in train mode so the
running mean/var track the target distribution, then saves updated checkpoints.

Usage (local, GPU1 only):
  CUDA_VISIBLE_DEVICES=1 python scripts/adabn_recompute.py \
    --garage-root /home/byeongjae/code/carla_garage \
    --base-dir  /home/byeongjae/code/checkpoints/transfuserpp/pretrained_models/all_towns \
    --out-dir   /home/byeongjae/code/checkpoints/transfuserpp/pretrained_models/all_towns_adabn \
    --index     /home/byeongjae/dataset/byeongjae/datasets/t2d_local_adabn_index.npz \
    --max-samples 2000
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from teach2drive_adapter.data import Teach2DriveIndexDataset
from teach2drive_adapter.transfuserpp_bridge import load_transfuserpp, prepare_transfuserpp_inputs


def _camera_list(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def _bn_modules(net):
    return [m for m in net.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))]


def recompute_one(garage_root, base_dir, ckpt_path, dataset, sample_ids, cameras, tfpp_camera, command_mode, device):
    net, config, load_info = load_transfuserpp(garage_root, base_dir, device=device, checkpoint=str(ckpt_path))
    net.eval()  # everything frozen/eval by default...
    bns = _bn_modules(net)
    for m in bns:
        m.reset_running_stats()          # running_mean=0, running_var=1, num_batches_tracked=0
        m.momentum = None                # cumulative moving average -> exact target-domain stats
        m.train()                        # ...only BN layers accumulate running stats
    batch_size = 16
    print(f"  [{ckpt_path.name}] BN modules={len(bns)} forwarding {len(sample_ids)} frames (bs={batch_size})", flush=True)
    with torch.no_grad():
        done = 0
        for start in range(0, len(sample_ids), batch_size):
            chunk = sample_ids[start:start + batch_size]
            if len(chunk) < 2:  # BN1d in train mode needs >1 sample per channel
                continue
            samples = [dataset[sid] for sid in chunk]
            scalar = torch.stack([s["scalar"] for s in samples]).to(device=device, dtype=torch.float32)
            camera = torch.stack([s["camera"] for s in samples]).to(device=device, dtype=torch.float32)
            lidar = torch.stack([s["lidar"] for s in samples]).to(device=device, dtype=torch.float32)
            inputs = prepare_transfuserpp_inputs(
                scalar=scalar, camera=camera, lidar=lidar, cameras=cameras,
                config=config, command_mode=command_mode, tfpp_camera=tfpp_camera,
            )
            net(**inputs)
            done += len(chunk)
            if done <= batch_size or done % 256 < batch_size:
                print(f"    forwarded {done}/{len(sample_ids)}", flush=True)
    net.eval()
    return net.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--garage-root", required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--cameras", default="left,front,right")
    ap.add_argument("--tfpp-camera", default="front")
    ap.add_argument("--command-mode", choices=["lane_follow", "target_angle"], default="lane_follow")
    ap.add_argument("--image-size", type=int, nargs=2, default=[640, 360])
    ap.add_argument("--lidar-size", type=int, default=128)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--episode-root-override", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cameras = _camera_list(args.cameras)
    base_dir = Path(args.base_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Teach2DriveIndexDataset(
        args.index, cameras=cameras, image_size=tuple(args.image_size),
        lidar_size=args.lidar_size, episode_root_override=args.episode_root_override,
    )
    total = len(dataset)
    if args.max_samples > 0 and args.max_samples < total:
        stride = max(1, total // args.max_samples)
        sample_ids = list(range(0, total, stride))[: args.max_samples]
    else:
        sample_ids = list(range(total))
    print(f"[adabn] device={device} dataset={total} frames, using {len(sample_ids)} for BN", flush=True)

    ckpts = sorted(base_dir.glob("model_*.pth"))
    print(f"[adabn] {len(ckpts)} base checkpoints: {[c.name for c in ckpts]}", flush=True)
    for ckpt in ckpts:
        sd = recompute_one(args.garage_root, str(base_dir), ckpt, dataset, sample_ids,
                           cameras, args.tfpp_camera, args.command_mode, device)
        torch.save(sd, out_dir / ckpt.name)
        print(f"  saved {out_dir / ckpt.name}", flush=True)

    # copy non-weight config files (config.json, args.txt, etc.) so the dir is a drop-in TEAM_CONFIG
    for f in base_dir.iterdir():
        if f.is_file() and not f.name.endswith(".pth"):
            shutil.copy2(f, out_dir / f.name)
    print(f"[adabn] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
