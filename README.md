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

