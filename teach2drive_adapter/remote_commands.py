import argparse
from textwrap import dedent


def _csv_gpus(raw: str) -> str:
    raw = raw.strip()
    if raw.lower() in {"all", ""}:
        return "0,1,2,3,4,5,6,7"
    return raw


def _section(title: str, body: str) -> str:
    return f"\n# {'=' * 78}\n# {title}\n# {'=' * 78}\n{dedent(body).strip()}\n"


def docker_block(args: argparse.Namespace) -> str:
    return _section(
        "Host shell: create or enter Docker container",
        f"""
        # Create once. If the container already exists, use the docker start command below.
        docker run -it --name {args.docker_name} --shm-size={args.shm_size} \\
          -v {args.host_workspace}:/workspace \\
          -v {args.host_datasets}:/datasets \\
          --gpus all \\
          {args.image} \\
          /bin/bash

        # Re-enter later.
        docker start -ai {args.docker_name}
        """,
    )


def setup_block(args: argparse.Namespace) -> str:
    return _section(
        "Inside Docker: install repo and Python dependencies",
        f"""
        set -euo pipefail

        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y \\
          git curl wget unzip ca-certificates libgl1 libglib2.0-0

        mkdir -p {args.workspace}
        cd {args.workspace}

        if [ ! -d teach2drive_adapter/.git ]; then
          git clone {args.repo_url} teach2drive_adapter
        fi

        cd teach2drive_adapter
        git pull --ff-only

        python -m pip install --upgrade pip
        python -m pip install -e .
        python -m pip install opencv-python-headless numpy tqdm

        python - <<'PY'
        import torch
        print("torch", torch.__version__)
        print("cuda_available", torch.cuda.is_available())
        print("gpu_count", torch.cuda.device_count())
        PY
        """,
    )


def dataset_block(args: argparse.Namespace) -> str:
    return _section(
        "Inside Docker: dataset and index sanity check",
        f"""
        set -euo pipefail
        cd {args.repo_dir}

        echo "DATA_ROOT={args.dataset_root}"
        du -shL {args.dataset_root} || true
        find -L {args.dataset_root} -name frames.jsonl | wc -l
        find -L {args.dataset_root} -name pseudo_labels_multicam.jsonl | wc -l

        echo "INDEX={args.index}"
        ls -lh {args.index}

        python - <<'PY'
        import numpy as np
        p = "{args.index}"
        d = np.load(p, allow_pickle=True)
        print("samples", len(d["sample_indices"]) if "sample_indices" in d.files else "unknown")
        print("episodes", len(d["episode_dirs"]) if "episode_dirs" in d.files else "unknown")
        print("first_episode", d["episode_dirs"][0] if "episode_dirs" in d.files else "unknown")
        PY
        """,
    )


def layout_block(args: argparse.Namespace) -> str:
    return _section(
        "Inside Docker: write canonical sensor_layout.json sidecars",
        f"""
        set -euo pipefail
        cd {args.repo_dir}

        python -m teach2drive_adapter.write_sensor_layouts \\
          --input-root {args.dataset_root} \\
          --output-name sensor_layout.json

        find -L {args.dataset_root} -name sensor_layout.json | wc -l
        """,
    )


def tfpp_install_block(args: argparse.Namespace) -> str:
    return _section(
        "Inside Docker: install CARLA Garage TransFuser++ code and pretrained weights",
        f"""
        set -euo pipefail
        cd {args.repo_dir}

        ROOT={args.workspace} \\
        GARAGE_DIR={args.tfpp_root} \\
        WEIGHT_DIR={args.tfpp_weight_dir} \\
        CARLA_ROOT={args.carla_root} \\
        bash scripts/install_transfuserpp.sh

        find {args.tfpp_weight_dir} -maxdepth 4 -type f \\
          \\( -name '*.pth' -o -name '*.pt' -o -name 'config.json' -o -name 'args.txt' \\) | sort | sed -n '1,80p'
        """,
    )


def transfuser2022_train_block(args: argparse.Namespace) -> str:
    max_train = f"          --max-train-samples {args.max_train_samples} \\\n" if args.max_train_samples > 0 else ""
    max_val = f"          --max-val-samples {args.max_val_samples} \\\n" if args.max_val_samples > 0 else ""
    return _section(
        "Inside Docker: train frozen original TransFuser adapter baseline",
        f"""
        set -euo pipefail
        cd {args.repo_dir}

        mkdir -p {args.out_dir}

        PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES={_csv_gpus(args.gpus)} python -m teach2drive_adapter.train_transfuser_adapter \\
          --index {args.index} \\
          --out-dir {args.out_dir} \\
          --transfuser-root {args.transfuser_root} \\
          --team-config {args.team_config} \\
          --episode-root-override {args.dataset_root} \\
          --cameras left,front,right \\
          --epochs {args.epochs} \\
          --batch-size {args.batch_size} \\
          --num-workers {args.num_workers} \\
          --data-parallel \\
{max_train}{max_val}          --lr {args.lr} \\
          --weight-decay {args.weight_decay} \\
          --xy-loss-weight {args.xy_loss_weight} \\
          --yaw-loss-weight {args.yaw_loss_weight} \\
          --speed-loss-weight {args.speed_loss_weight} \\
          --stop-loss-weight {args.stop_loss_weight} \\
          --stop-state-loss-weight {args.stop_state_loss_weight} \\
          --stop-reason-loss-weight {args.stop_reason_loss_weight} \\
          --grad-clip {args.grad_clip} \\
          --step-log-every {args.step_log_every} \\
          2>&1 | tee {args.out_dir}/train.log
        """,
    )


def note_block(_args: argparse.Namespace) -> str:
    return _section(
        "Current limitation",
        """
        # TransFuser++ layout-adapter training is the next implementation target.
        # The current runnable training command above is the original TransFuser 2022
        # frozen-backbone adapter baseline. Use the TransFuser++ install block now so
        # the server is ready, then pull the next commit when the TF++ bridge lands.
        """,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print copy-paste remote training commands for Teach2Drive.")
    parser.add_argument("--mode", choices=["all", "docker", "setup", "check", "layout", "tfpp-install", "train2022"], default="all")
    parser.add_argument("--repo-url", default="https://github.com/Donok53/teach2drive_adapter.git")
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--repo-dir", default="/workspace/teach2drive_adapter")
    parser.add_argument("--host-workspace", default="/data/teach2drive/workspace")
    parser.add_argument("--host-datasets", default="/data/teach2drive/datasets")
    parser.add_argument("--docker-name", default="teach2drive_a100")
    parser.add_argument("--image", default="pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel")
    parser.add_argument("--shm-size", default="64g")
    parser.add_argument("--dataset-root", default="/datasets/teach2drive/town10_3cam_640x360")
    parser.add_argument("--index", default="/workspace/teach2drive_bootstrap/runs/town10_3cam_640x360_tokens/token_pseudo_rule_multicam_index.npz")
    parser.add_argument("--tfpp-root", default="/workspace/carla_garage")
    parser.add_argument("--tfpp-weight-dir", default="/workspace/checkpoints/transfuserpp")
    parser.add_argument("--carla-root", default="/workspace/CARLA_0.9.15")
    parser.add_argument("--transfuser-root", default="/workspace/transfuser")
    parser.add_argument("--team-config", default="/workspace/transfuser/model_ckpt/models_2022/transfuser")
    parser.add_argument("--out-dir", default="runs/transfuser2022_adapter_a100")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=4096)
    parser.add_argument("--max-val-samples", type=int, default=512)
    parser.add_argument("--lr", default="3e-5")
    parser.add_argument("--weight-decay", default="1e-4")
    parser.add_argument("--xy-loss-weight", default="1.0")
    parser.add_argument("--yaw-loss-weight", default="0.20")
    parser.add_argument("--speed-loss-weight", default="0.20")
    parser.add_argument("--stop-loss-weight", default="0.05")
    parser.add_argument("--stop-state-loss-weight", default="0.10")
    parser.add_argument("--stop-reason-loss-weight", default="0.02")
    parser.add_argument("--grad-clip", default="1.0")
    parser.add_argument("--step-log-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    blocks = {
        "docker": docker_block,
        "setup": setup_block,
        "check": dataset_block,
        "layout": layout_block,
        "tfpp-install": tfpp_install_block,
        "train2022": transfuser2022_train_block,
    }
    if args.mode == "all":
        order = ["docker", "setup", "check", "layout", "tfpp-install", "train2022"]
    else:
        order = [args.mode]
    print(dedent("# Teach2Drive remote command sheet\n# Copy each block to the server in order.\n"))
    for name in order:
        print(blocks[name](args))
    if args.mode in {"all", "tfpp-install"}:
        print(note_block(args))


if __name__ == "__main__":
    main()
