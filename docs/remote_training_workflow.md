# Remote Training Workflow

The remote A100 server only needs to execute commands. Codex stays local and
prints command blocks with:

```bash
cd ~/code/teach2drive_adapter
python -m teach2drive_adapter.remote_commands --mode all
```

Use this when the server cannot run Codex directly.

## Common Local Command Generator

Print every block:

```bash
python -m teach2drive_adapter.remote_commands \
  --mode all \
  --host-workspace /data/teach2drive/workspace \
  --host-datasets /data/teach2drive/datasets \
  --dataset-root /datasets/teach2drive/town10_3cam_640x360 \
  --index /workspace/teach2drive_bootstrap/runs/town10_3cam_640x360_tokens/token_pseudo_rule_multicam_index.npz \
  --gpus 0,1,2,3,4,5,6,7
```

Print one block at a time:

```bash
python -m teach2drive_adapter.remote_commands --mode docker
python -m teach2drive_adapter.remote_commands --mode setup
python -m teach2drive_adapter.remote_commands --mode check
python -m teach2drive_adapter.remote_commands --mode layout
python -m teach2drive_adapter.remote_commands --mode tfpp-install
python -m teach2drive_adapter.remote_commands --mode train2022
```

## What Runs Today

The current runnable training command is:

```text
frozen original TransFuser 2022 + Teach2Drive residual adapter
```

This is still useful as a baseline.

## What Is Being Prepared

The TransFuser++ install block prepares:

```text
/workspace/carla_garage
/workspace/checkpoints/transfuserpp
```

The actual TransFuser++ sensor-layout adapter training command will be added
after the bridge from Teach2Drive tensors to CARLA Garage TransFuser++ tensors is
implemented.

## Server Rule

Do not edit code manually on the server. Use:

```bash
cd /workspace/teach2drive_adapter
git pull --ff-only
```

Then run the command block produced locally.
