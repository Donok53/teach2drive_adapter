# TransFuser++ Sensor-Layout Adapter Plan

## Goal

The next Teach2Drive direction is not "train a new end-to-end model from a
small log."  The goal is:

```text
pretrained TransFuser++ driving prior
  + small user dataset
  + explicit sensor pose/layout metadata
  -> adapted policy for a changed robot/vehicle sensor rig
```

The first research target is sensor pose/layout change while keeping the sensor
types mostly fixed:

- camera count: start with 3 cameras, then ablate to front-only
- LiDAR: present, same BEV representation at first
- perturbations: camera x/y/z, yaw, pitch, FOV, crop, and LiDAR extrinsics

This is narrower than "any sensor in the world", but it is publishable because
it attacks a concrete brittleness in pretrained E2E driving models.

## Why TransFuser++

CARLA Garage provides TransFuser++ code and official pretrained model folders.
It uses CARLA 0.9.15, supports CARLA Leaderboard 2.0 and Bench2Drive-style
closed-loop evaluation, and the released model is a strong open-source driving
prior.

Official setup facts:

- repository: `https://github.com/autonomousvision/carla_garage`
- branch: `leaderboard_2`
- pretrained model archive: about 2.6 GB
- official training data archive: about 364 GB
- full training is expensive, roughly days on multi-A100 nodes, so this project
  should train adapters first.

## Installation

Local install/checkpoint download:

```bash
cd ~/code/teach2drive_adapter
bash configs/transfuserpp_install_local.sh
```

Default locations:

```text
~/code/carla_garage
~/code/checkpoints/transfuserpp
~/carla-simulator
```

On a remote A100 node, keep the same structure if possible:

```bash
ROOT=/workspace \
GARAGE_DIR=/workspace/carla_garage \
WEIGHT_DIR=/workspace/checkpoints/transfuserpp \
CARLA_ROOT=/workspace/CARLA_0.9.15 \
bash scripts/install_transfuserpp.sh
```

## Code Direction

### Stage 0: Compatibility Check

Before training an adapter, verify that our Teach2Drive dataset can be converted
to the exact tensors expected by TransFuser++:

- RGB camera tensors in the expected camera order
- LiDAR BEV or raw LiDAR converted to the expected representation
- speed and route target point
- sensor metadata sidecar loaded per episode

Output:

```text
runs/tfpp_bridge/audit.json
runs/tfpp_bridge/audit_preview.jpg
```

### Stage 1: Frozen Backbone, Action Adapter

Baseline:

```text
sensor data -> frozen TransFuser++ -> fused feature -> small action adapter
```

This tells us whether the pretrained prior already helps more than scratch
training.

### Stage 2: Frozen Backbone, Sensor-Layout Adapter

Main research track:

```text
sensor data + sensor extrinsics/FOV
  -> layout encoder
  -> feature modulation / residual correction
  -> frozen TransFuser++ fused feature
  -> action adapter
```

The adapter should be small enough to fine-tune from short logs.

### Stage 3: Sensor Pose Perturbation Benchmark

Create controlled CARLA variants:

| Split | Meaning |
| --- | --- |
| canonical | original TransFuser++-like sensor layout |
| shifted-camera-yaw | camera yaw offsets |
| shifted-camera-height | camera height/pitch offsets |
| shifted-lidar | LiDAR x/y/z/yaw offsets |
| mixed-layout | random combination of small pose shifts |

Compare:

| Method | Trainable parts |
| --- | --- |
| zero-shot TF++ | none |
| action adapter only | output adapter |
| layout adapter | sensor-layout + output adapter |
| partial fine-tune | late fusion/head |
| full fine-tune | all parameters, upper bound only |

The expected claim is not that adapters beat full fine-tuning with unlimited
data. The expected claim is that layout-aware adapters recover more performance
than ordinary fine-tuning when data is small.

## Data Format Extension

Each episode should optionally contain:

```text
sensor_layout.json
```

Example:

```json
{
  "cameras": {
    "front": {"x": 1.3, "y": 0.0, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "fov": 120.0},
    "left": {"x": 1.3, "y": -0.4, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": -60.0, "fov": 120.0},
    "right": {"x": 1.3, "y": 0.4, "z": 2.3, "roll": 0.0, "pitch": 0.0, "yaw": 60.0, "fov": 120.0}
  },
  "lidars": {
    "top": {"x": 1.3, "y": 0.0, "z": 2.5, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "range": 85.0}
  }
}
```

If no file exists, the loader should fall back to the canonical TransFuser++
layout and mark the layout as estimated.

Write canonical sidecars for an existing Teach2Drive dataset:

```bash
cd ~/code/teach2drive_adapter
PY=/home/byeongjae/miniconda3/envs/vad/bin/python \
DATA_ROOT=~/code/teach2drive_bootstrap/data/carla/town10_3cam_640x360 \
bash configs/write_canonical_sensor_layouts.sh
```

Create a shifted-layout copy for a controlled experiment by writing a different
sidecar name:

```bash
$PY -m teach2drive_adapter.write_sensor_layouts \
  --input-root ~/code/teach2drive_bootstrap/data/carla/town10_3cam_640x360 \
  --output-name sensor_layout_camera_yaw_p10.json \
  --camera-yaw-deg 10 \
  --overwrite
```

## First Implementation Milestones

1. Add `sensor_layout.json` loader and layout embedding.
2. Build `TransFuserPPBridge` that loads CARLA Garage model folders.
3. Train `TFPPActionAdapter` on canonical data only.
4. Generate small pose-shift CARLA datasets.
5. Train `TFPPLayoutAdapter` and compare against action-only adapter.
