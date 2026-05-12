# Teach2Drive Adapter

This folder is the second-stage research codebase for Teach2Drive.

The old `teach2drive_bootstrap` project answers:

```text
Can we train a small end-to-end policy from a short user driving log?
```

The answer from the CARLA experiments was mostly no: scratch training from a
small dataset causes stop-go behavior, weak stop-sign/light compliance, and
closed-loop drift.

This project answers the next question:

```text
Can a pretrained end-to-end policy be adapted to a new robot, vehicle, or sensor
layout with a short user driving log?
```

The intended structure is:

```text
bag/log/CARLA episode
  -> Teach2Drive unified token index
  -> frozen or pretrained sensor backbone
  -> small adapter
  -> device-specific action/planning head
```

## What Is Implemented

- A dataset loader for the existing Teach2Drive token index `.npz`.
- A portable sensor backbone for camera, LiDAR BEV, and scalar odom/IMU tokens.
- A bottleneck adapter that can be fine-tuned while the backbone is frozen.
- Training modes:
  - `scratch`: train everything.
  - `adapter`: freeze sensor backbone, train adapter and heads.
  - `head`: freeze backbone and adapter, train heads only.
  - `full`: train all parameters from a pretrained checkpoint.
- Open-loop validation metrics for trajectory, speed, stop, stop-state, and
  stop-reason heads.

This is intentionally a clean scaffold. The next step is to replace or initialize
the `PortableBackbone` with a stronger pretrained E2E model such as TransFuser,
TCP, InterFuser, or a robotics foundation model.

## TransFuser Bridge

The first pretrained E2E target is
[TransFuser](https://github.com/autonomousvision/transfuser). Its official
agent expects:

- RGB stitched in left/front/right order, cropped to `3 x 160 x 704`.
- LiDAR as a 2-channel `256 x 256` histogram generated from raw point clouds.
- Speed plus a local route target point.

Teach2Drive can audit whether a collected dataset can be converted into that
shape:

```bash
PY=/home/byeongjae/miniconda3/envs/vad/bin/python

$PY -m teach2drive_adapter.audit_transfuser_bridge \
  --index ~/code/teach2drive_bootstrap/runs/town10_3cam_640x360_tokens/token_pseudo_rule_multicam_index.npz \
  --sample-index 1000 \
  --output runs/transfuser_bridge/audit_3cam.json \
  --preview runs/transfuser_bridge/audit_3cam.jpg
```

To simulate a robot with only a front camera, keep TransFuser's expected shape
while explicitly marking the missing-camera strategy:

```bash
$PY -m teach2drive_adapter.audit_transfuser_bridge \
  --index ~/code/teach2drive_bootstrap/runs/town10_3cam_640x360_tokens/token_pseudo_rule_multicam_index.npz \
  --available-cameras front \
  --missing-camera-policy repeat_front \
  --sample-index 1000 \
  --output runs/transfuser_bridge/audit_front_only.json \
  --preview runs/transfuser_bridge/audit_front_only.jpg
```

This bridge is not yet the final pretrained adapter. It is the compatibility
layer used before loading TransFuser weights.

After downloading TransFuser's official model folder, run a zero-shot open-loop
check before fitting any adapter:

```bash
$PY -m teach2drive_adapter.eval_transfuser_openloop \
  --index ~/code/teach2drive_bootstrap/runs/town10_3cam_640x360_tokens/token_pseudo_rule_multicam_index.npz \
  --transfuser-root ~/code/transfuser \
  --team-config ~/code/transfuser/model_ckpt/models_2022/transfuser \
  --out-dir runs/transfuser_zero_shot \
  --max-samples 256
```

This reports waypoint error before adaptation. Adapter training should be judged
against this zero-shot score and the existing scratch baselines.

To train a small residual Teach2Drive adapter on top of frozen TransFuser
weights:

```bash
$PY -m teach2drive_adapter.train_transfuser_adapter \
  --index ~/code/teach2drive_bootstrap/runs/town10_3cam_640x360_tokens/token_pseudo_rule_multicam_index.npz \
  --out-dir runs/transfuser_adapter_v3_random_xy_small \
  --transfuser-root ~/code/transfuser \
  --team-config ~/code/transfuser/model_ckpt/models_2022/transfuser \
  --epochs 3 \
  --batch-size 8 \
  --num-workers 4 \
  --max-train-samples 4096 \
  --max-val-samples 512 \
  --lr 3e-5 \
  --xy-loss-weight 1.0 \
  --yaw-loss-weight 0.05 \
  --speed-loss-weight 0.02
```

The adapter checkpoint stores only Teach2Drive adapter/head weights, not the
full TransFuser checkpoint.

## TransFuser++ Layout Adapter Track

The next research track uses
[CARLA Garage TransFuser++](https://github.com/autonomousvision/carla_garage)
as the stronger pretrained driving prior and adds explicit sensor layout
conditioning.

Install the CARLA Garage code and official TransFuser++ model folders:

```bash
cd ~/code/teach2drive_adapter
bash configs/transfuserpp_install_local.sh
```

This downloads the `leaderboard_2` branch and the official pretrained model
archive into:

```text
~/code/carla_garage
~/code/checkpoints/transfuserpp
```

The current design note is in:

```text
docs/transfuserpp_layout_adapter_plan.md
```

The first code pieces for sensor metadata are:

```text
teach2drive_adapter/sensor_layout.py
teach2drive_adapter/layout_conditioning.py
```

### Collect TransFuser++-Style CARLA Data

After starting a CARLA 0.9.15 server, collect a 2-3 hour layout-adaptation
dataset with:

```bash
cd ~/code/teach2drive_adapter

CARLA_ROOT=~/dataset/byeongjae/carla-simulator \
OUTPUT_ROOT=~/dataset/byeongjae/t2d_transfuserpp_2h \
bash configs/collect_transfuserpp_2h.sh
```

The collector writes two sensor-layout profiles by default:

| Profile | Sensors |
| --- | --- |
| `tfpp_ego` | Official TransFuser++ front RGB camera at `1024x512`, `110 deg` FOV, plus roof LiDAR. |
| `front_triplet_shifted` | Three front-facing RGB cameras named `left/front/right` with small pose offsets, plus a slightly shifted LiDAR. |

Both profiles use the TransFuser++ CARLA rate conventions: simulator at `20 Hz`,
`data_save_freq=5` (`4` saved frames per second), camera `1024x512`, FOV `110`,
LiDAR `600000` points/sec and `10 Hz` rotation. Each episode stores:

```text
episode_xxxxxx/
  camera/{front,left,right}/0000.jpg
  rgb/0000.jpg
  lidar/0000.npz
  lidar_bev/0000.npy
  measurements/0000.json.gz
  sensor_layout.json
  frames.jsonl
```

`measurements/*.json.gz` contains the fields used by TransFuser++ inference and
training adapters: speed, target speed, target point, next target point, route,
command, next command, controls, IMU, ego pose, and route angle.

By default LiDAR is stored as compressed NumPy point clouds (`.npz`) plus a
TransFuser++-shape BEV histogram. If you need CARLA Garage-style `.laz` files,
install `laspy`/`lazrs` and run with `LIDAR_FORMAT=both`.

This track should compare:

| Method | Purpose |
| --- | --- |
| zero-shot TF++ | How brittle the pretrained model is under a new layout. |
| action adapter only | Whether output correction alone is enough. |
| layout adapter | Whether camera/LiDAR pose metadata improves small-data adaptation. |
| partial/full fine-tune | Upper-bound comparison when compute and data are enough. |

## Quick Start

Use an index produced by `teach2drive_bootstrap`:

```bash
cd ~/code/teach2drive_adapter

PY=/home/byeongjae/miniconda3/envs/vad/bin/python

$PY -m teach2drive_adapter.train_adapter \
  --index ~/code/teach2drive_bootstrap/runs/town10_3cam_640x360_tokens/token_pseudo_rule_multicam_index.npz \
  --out-dir runs/scratch_debug \
  --mode scratch \
  --epochs 2 \
  --batch-size 8 \
  --num-workers 2 \
  --image-size 160 90
```

Adapter fine-tuning from a checkpoint:

```bash
$PY -m teach2drive_adapter.train_adapter \
  --index ~/code/teach2drive_bootstrap/runs/town10_3cam_640x360_tokens/token_pseudo_rule_multicam_index.npz \
  --out-dir runs/adapter_v1 \
  --pretrained runs/pretrain/best_model.pt \
  --mode adapter \
  --epochs 20 \
  --batch-size 32 \
  --num-workers 8 \
  --data-parallel
```

If an index contains old absolute episode paths, use `--episode-root-override`:

```bash
--episode-root-override /fastdata/teach2drive/town10_3cam_640x360
```

## Research Comparison

Recommended experiments:

| Track | Description |
| --- | --- |
| Scratch | Train the whole model from short data only. |
| Full FT | Load a pretrained checkpoint and fine-tune all parameters. |
| Adapter FT | Freeze the backbone and train only adapter + heads. |
| Head FT | Freeze backbone + adapter and train only action heads. |

The expected claim is not that scratch learning works from tiny data. The claim
is that a pretrained driving/robotics prior plus a small Teach2Drive adapter
improves small-data deployment.
