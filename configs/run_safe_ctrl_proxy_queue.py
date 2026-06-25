#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import time
from dataclasses import dataclass


ADAPTER_ROOT = pathlib.Path(os.environ.get("ADAPTER_ROOT", "/home/jovyan/teach2drive/workspace/teach2drive_adapter_pdm_d_20260606"))
PY = os.environ.get("PY", "/home/jovyan_venv/.venv/torch2.1.2-py3.10-cuda11.8/bin/python")
SOURCE_DATA_ROOT = os.environ.get("SOURCE_DATA_ROOT", "/home/jovyan/dataset/byeongjae/datasets/t2d_tesla_benchmix8_front_triplet_target_3h")
PREP_ROOT = pathlib.Path(os.environ.get("PREP_ROOT", "/home/jovyan/dataset/byeongjae/runs/tfpp_tesla_benchmix8_front_triplet_target_3h_prepared"))
PROFILE = os.environ.get("PROFILE", "front_triplet_shifted")
TARGET_VIEW = pathlib.Path(os.environ.get("TARGET_VIEW", str(PREP_ROOT / "profile_views" / PROFILE)))
TARGET_INDEX = pathlib.Path(os.environ.get("TARGET_INDEX", str(PREP_ROOT / "indexes" / f"{PROFILE}_index.npz")))
INIT_CHECKPOINT = pathlib.Path(os.environ.get("INIT_CHECKPOINT", "/home/jovyan/dataset/byeongjae/runs/benchmix8_pdm_d_safety_recipe/pdm_d_topk_safe_select_lr1e6_b24/best_model.pt"))
RUN_ROOT = pathlib.Path(os.environ.get("RUN_ROOT", "/home/jovyan/dataset/byeongjae/runs/benchmix8_safe_ctrl_conservative_queue_target_only"))
LOG_DIR = pathlib.Path(os.environ.get("LOG_DIR", "/home/jovyan/teach2drive/logs/benchmix8_safe_ctrl_conservative_queue_target_only"))
BASELINE_SAFE_CTRL = float(os.environ.get("SAFE_CTRL_BASELINE", "7.272826818413129"))
THRESHOLD = float(os.environ.get("SAFE_CTRL_THRESHOLD", "6.2"))
EPOCH1_THRESHOLD = float(os.environ.get("SAFE_CTRL_EPOCH1_THRESHOLD", "6.8"))
EPOCH2_THRESHOLD = float(os.environ.get("SAFE_CTRL_EPOCH2_THRESHOLD", str(THRESHOLD)))
PROMOTE_EPOCHS = int(os.environ.get("PROMOTE_EPOCHS", "10"))
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "30"))
GPU_IDS = [int(x) for x in os.environ.get("SAFE_CTRL_GPUS", "0,2").replace(" ", "").split(",") if x]
FREE_MEMORY_MAX_MB = int(os.environ.get("FREE_MEMORY_MAX_MB", "2500"))
RECIPE_SET = os.environ.get("SAFE_CTRL_RECIPE_SET", "conservative").strip().lower()
ONLY_RECIPES = {x.strip() for x in os.environ.get("SAFE_CTRL_ONLY_RECIPES", "").split(",") if x.strip()}
SKIP_RECIPES = {x.strip() for x in os.environ.get("SAFE_CTRL_SKIP_RECIPES", "").split(",") if x.strip()}


@dataclass(frozen=True)
class Recipe:
    name: str
    env: dict[str, str]


def s(value: object) -> str:
    return str(value)


COMMON = {
    "PY": PY,
    "PYTHONUNBUFFERED": "1",
    "SOURCE_DATA_ROOT": SOURCE_DATA_ROOT,
    "SNAPSHOT_COMPLETE_EPISODES": "0",
    "REFRESH_SNAPSHOT": "0",
    "SKIP_EXPORT": "1",
    "INDEX_OVERWRITE": "0",
    "TARGET_VIEW": str(TARGET_VIEW),
    "TARGET_INDEX": str(TARGET_INDEX),
    "OVERWRITE": "1",
    "SAVE_EPOCH_CHECKPOINTS": "1",
    "EPOCH_CHECKPOINT_DIR": "epoch_checkpoints",
    "NUM_WORKERS": "4",
    "EPOCHS": s(PROMOTE_EPOCHS),
    "EARLY_STOP_PATIENCE": "4",
    "EARLY_STOP_MIN_DELTA": "0.0",
    "SPLIT_MODE": "reason_coverage",
    "VAL_REQUIRED_STOP_REASONS": "traffic_light,stop_sign,front_vehicle,junction_yield",
    "VAL_MIN_REASON_SAMPLES": "32",
    "SEED": "931",
    "SELECTION_METRIC": "safety_controller_closed_loop_proxy",
    "SELECTION_MODE": "min",
    "INIT_CHECKPOINT": str(INIT_CHECKPOINT),
    "EXTRINSIC_AWARE": "1",
    "SOURCE_PROFILE": "front_triplet_shifted",
    "STAGE_ADAPTER_MODALITIES": "all",
    "FUSION_ADAPTER_ENABLED": "1",
    "HAZARD_STOP_REASONS": "traffic_light,stop_sign,front_vehicle,junction_yield",
    "OUTPUT_RESIDUAL": "1",
    "OUTPUT_RESIDUAL_HIDDEN_DIM": "512",
    "OUTPUT_RESIDUAL_DROPOUT": "0.03",
    "YAW_LOSS_WEIGHT": "0.01",
    "TRAJ_SMOOTH_LOSS_WEIGHT": "0.025",
    "SPEED_SMOOTH_LOSS_WEIGHT": "0.018",
    "STOP_SPEED_CEILING_MPS": "0.35",
    "STOP_PROGRESS_CEILING_M": "0.65",
    "CLOSED_LOOP_UNROLL_STEPS": "6",
    "CLOSED_LOOP_UNROLL_DT": "0.5",
    "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.0",
}


BASE_RECIPES = [
    Recipe(
        "resid_stopsign_front_max_b24",
        {
            "BATCH_SIZE": "24",
            "LR": "5.0e-5",
            "WEIGHT_DECAY": "8e-5",
            "FREEZE_TASK_ADAPTER": "1",
            "LORA_RANK": "0",
            "UNFREEZE_INCLUDE": "",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "1.35",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "2.80",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-0.45",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.95",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.002",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.025",
            "XY_LOSS_WEIGHT": "0.12",
            "SPEED_LOSS_WEIGHT": "0.18",
            "MOVING_SAMPLE_WEIGHT": "0.60",
            "STOPPED_SAMPLE_WEIGHT": "3.80",
            "HAZARD_SAMPLE_WEIGHT": "14.00",
            "STOPSIGN_SAMPLE_WEIGHT": "6.00",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "3.50",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "3.00",
            "STOP_LOSS_WEIGHT": "0.28",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.26",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.25",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "3.00",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "2.00",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "3.20",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "3.40",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "2.80",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.010",
            "CONTROL_LOSS_WEIGHT": "1.20",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "1.50",
            "PDM_LATERAL_LOSS_WEIGHT": "0.25",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.02",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "2.20",
            "PDM_CONTROLLER_LOSS_WEIGHT": "2.80",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.45",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.35",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "3.20",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.00",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.15",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "6.0",
        },
    ),
    Recipe(
        "lora_controller_plan_b16",
        {
            "BATCH_SIZE": "16",
            "LR": "2.5e-5",
            "WEIGHT_DECAY": "1e-4",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "16",
            "LORA_ALPHA": "32.0",
            "LORA_DROPOUT": "0.03",
            "UNFREEZE_INCLUDE": "",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "1.05",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "2.10",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-0.8",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.85",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.004",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.040",
            "XY_LOSS_WEIGHT": "0.20",
            "SPEED_LOSS_WEIGHT": "0.26",
            "MOVING_SAMPLE_WEIGHT": "0.85",
            "STOPPED_SAMPLE_WEIGHT": "2.70",
            "HAZARD_SAMPLE_WEIGHT": "9.50",
            "STOPSIGN_SAMPLE_WEIGHT": "4.00",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "2.60",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "2.20",
            "STOP_LOSS_WEIGHT": "0.20",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.22",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.20",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "2.20",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.40",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "2.50",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "2.40",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "2.00",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.030",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.040",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.040",
            "CONTROL_LOSS_WEIGHT": "1.40",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "1.20",
            "PDM_LATERAL_LOSS_WEIGHT": "0.45",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.05",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "1.60",
            "PDM_CONTROLLER_LOSS_WEIGHT": "3.40",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "1.20",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "1.10",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "3.20",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "0.85",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.20",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "7.0",
        },
    ),
    Recipe(
        "balanced_go_safety_b24",
        {
            "BATCH_SIZE": "24",
            "LR": "3.2e-5",
            "WEIGHT_DECAY": "9e-5",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16.0",
            "LORA_DROPOUT": "0.02",
            "UNFREEZE_INCLUDE": "",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.95",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "1.90",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-0.9",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.75",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.006",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.055",
            "XY_LOSS_WEIGHT": "0.26",
            "SPEED_LOSS_WEIGHT": "0.34",
            "MOVING_SAMPLE_WEIGHT": "1.00",
            "STOPPED_SAMPLE_WEIGHT": "2.40",
            "HAZARD_SAMPLE_WEIGHT": "8.00",
            "STOPSIGN_SAMPLE_WEIGHT": "3.00",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "2.30",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "2.00",
            "STOP_LOSS_WEIGHT": "0.16",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.20",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.18",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "1.70",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.10",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.80",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.80",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.50",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.060",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.080",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.080",
            "CONTROL_LOSS_WEIGHT": "1.00",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "1.00",
            "PDM_LATERAL_LOSS_WEIGHT": "0.50",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.10",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "1.20",
            "PDM_CONTROLLER_LOSS_WEIGHT": "2.40",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.85",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.85",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "2.40",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "0.70",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.20",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "7.5",
        },
    ),
    Recipe(
        "unfreeze_speed_decoder_hazard_b16",
        {
            "BATCH_SIZE": "16",
            "LR": "1.8e-5",
            "WEIGHT_DECAY": "1e-4",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16.0",
            "LORA_DROPOUT": "0.02",
            "UNFREEZE_INCLUDE": "^target_speed_network\\.,^checkpoint_decoder\\.(encoder|decoder)\\.",
            "UNFREEZE_LR": "5.0e-6",
            "UNFREEZE_WEIGHT_DECAY": "5e-6",
            "INIT_PARAM_ANCHOR_LOSS_WEIGHT": "0.0003",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "1.00",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "2.40",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-0.7",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.85",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.004",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.040",
            "XY_LOSS_WEIGHT": "0.18",
            "SPEED_LOSS_WEIGHT": "0.24",
            "MOVING_SAMPLE_WEIGHT": "0.80",
            "STOPPED_SAMPLE_WEIGHT": "3.00",
            "HAZARD_SAMPLE_WEIGHT": "10.50",
            "TRAFFICLIGHT_SAMPLE_WEIGHT": "2.00",
            "STOPSIGN_SAMPLE_WEIGHT": "4.50",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "3.00",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "2.50",
            "STOP_LOSS_WEIGHT": "0.20",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.22",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.20",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "2.30",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.70",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "2.60",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "2.50",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "2.20",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.020",
            "CONTROL_LOSS_WEIGHT": "1.20",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "1.35",
            "PDM_LATERAL_LOSS_WEIGHT": "0.35",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.04",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "1.80",
            "PDM_CONTROLLER_LOSS_WEIGHT": "3.00",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.80",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.80",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "3.00",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "0.95",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.15",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "6.8",
        },
    ),
    Recipe(
        "front_junction_yield_boost_b24",
        {
            "BATCH_SIZE": "24",
            "LR": "3.8e-5",
            "WEIGHT_DECAY": "8e-5",
            "FREEZE_TASK_ADAPTER": "1",
            "LORA_RANK": "0",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "1.20",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "2.50",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-0.55",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.90",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.003",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.030",
            "XY_LOSS_WEIGHT": "0.12",
            "SPEED_LOSS_WEIGHT": "0.18",
            "MOVING_SAMPLE_WEIGHT": "0.65",
            "STOPPED_SAMPLE_WEIGHT": "3.40",
            "HAZARD_SAMPLE_WEIGHT": "12.00",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "5.50",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "5.00",
            "STOPSIGN_SAMPLE_WEIGHT": "2.00",
            "STOP_LOSS_WEIGHT": "0.26",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.24",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.24",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "2.60",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "3.80",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "3.50",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.60",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.40",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.012",
            "CONTROL_LOSS_WEIGHT": "1.00",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "1.40",
            "PDM_LATERAL_LOSS_WEIGHT": "0.25",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.02",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "2.50",
            "PDM_CONTROLLER_LOSS_WEIGHT": "2.60",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.40",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.30",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "2.90",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.05",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.10",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "6.2",
        },
    ),
    Recipe(
        "unroll_lateral_controller_b16",
        {
            "BATCH_SIZE": "16",
            "LR": "2.2e-5",
            "WEIGHT_DECAY": "1e-4",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "16",
            "LORA_ALPHA": "32.0",
            "LORA_DROPOUT": "0.03",
            "UNFREEZE_INCLUDE": "^join\\.layers\\.(4|5)\\.",
            "UNFREEZE_LR": "2.0e-6",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.95",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "2.00",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-0.85",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.80",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.005",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.050",
            "XY_LOSS_WEIGHT": "0.22",
            "SPEED_LOSS_WEIGHT": "0.24",
            "MOVING_SAMPLE_WEIGHT": "0.90",
            "STOPPED_SAMPLE_WEIGHT": "2.60",
            "HAZARD_SAMPLE_WEIGHT": "9.00",
            "STOPSIGN_SAMPLE_WEIGHT": "3.50",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "2.80",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "2.50",
            "STOP_LOSS_WEIGHT": "0.18",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.22",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.18",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "2.00",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "2.20",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "2.20",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "2.00",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.025",
            "CONTROL_LOSS_WEIGHT": "1.10",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "1.10",
            "PDM_LATERAL_LOSS_WEIGHT": "1.20",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.04",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "1.60",
            "PDM_CONTROLLER_LOSS_WEIGHT": "2.70",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "1.30",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.80",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "2.70",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.40",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.35",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "7.0",
        },
    ),
]


HINGE_RECIPES = [
    Recipe(
        "hinge_go_progress_lora_b16",
        {
            "BATCH_SIZE": "16",
            "LR": "2.8e-5",
            "WEIGHT_DECAY": "9e-5",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "16",
            "LORA_ALPHA": "32.0",
            "LORA_DROPOUT": "0.03",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.90",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "1.80",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-0.90",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.75",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.006",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.045",
            "XY_LOSS_WEIGHT": "0.22",
            "SPEED_LOSS_WEIGHT": "0.30",
            "MOVING_SAMPLE_WEIGHT": "1.35",
            "STOPPED_SAMPLE_WEIGHT": "1.80",
            "HAZARD_SAMPLE_WEIGHT": "5.50",
            "STOPSIGN_SAMPLE_WEIGHT": "2.50",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "1.80",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "1.80",
            "STOP_LOSS_WEIGHT": "0.12",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.14",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.12",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "1.15",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "0.80",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.20",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.10",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.00",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.080",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.140",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.140",
            "CONTROL_LOSS_WEIGHT": "0.80",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.70",
            "PDM_LATERAL_LOSS_WEIGHT": "0.45",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.45",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.75",
            "PDM_CONTROLLER_LOSS_WEIGHT": "1.40",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.70",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.60",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "1.20",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "6.00",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "3.50",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "2.50",
            "CONTROLLER_GO_HINGE_MARGIN": "0.05",
            "CONTROLLER_SOFT_STOP_TEMPERATURE": "0.75",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.00",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.20",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "7.5",
        },
    ),
    Recipe(
        "hinge_closed_progress_unfreeze_b16",
        {
            "BATCH_SIZE": "16",
            "LR": "1.8e-5",
            "WEIGHT_DECAY": "9e-5",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16.0",
            "LORA_DROPOUT": "0.02",
            "UNFREEZE_INCLUDE": "^target_speed_network\\.,^checkpoint_decoder\\.(encoder|decoder)\\.",
            "UNFREEZE_LR": "5.0e-6",
            "UNFREEZE_WEIGHT_DECAY": "5e-6",
            "INIT_PARAM_ANCHOR_LOSS_WEIGHT": "0.0003",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.95",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "2.00",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-0.80",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.80",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.005",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.045",
            "XY_LOSS_WEIGHT": "0.24",
            "SPEED_LOSS_WEIGHT": "0.32",
            "MOVING_SAMPLE_WEIGHT": "1.25",
            "STOPPED_SAMPLE_WEIGHT": "2.10",
            "HAZARD_SAMPLE_WEIGHT": "6.00",
            "STOPSIGN_SAMPLE_WEIGHT": "2.80",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "2.00",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "2.00",
            "STOP_LOSS_WEIGHT": "0.14",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.16",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.14",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "1.30",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "0.90",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.35",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.25",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.10",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.090",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.160",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.160",
            "CONTROL_LOSS_WEIGHT": "0.90",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.75",
            "PDM_LATERAL_LOSS_WEIGHT": "0.55",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.65",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.85",
            "PDM_CONTROLLER_LOSS_WEIGHT": "1.55",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.85",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.70",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "1.25",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "5.50",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "3.00",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "4.00",
            "CONTROLLER_GO_HINGE_MARGIN": "0.06",
            "CONTROLLER_SOFT_STOP_TEMPERATURE": "0.70",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.30",
            "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.15",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.25",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "8.0",
        },
    ),
    Recipe(
        "hinge_threshold_residual_b24",
        {
            "BATCH_SIZE": "24",
            "LR": "4.0e-5",
            "WEIGHT_DECAY": "8e-5",
            "FREEZE_TASK_ADAPTER": "1",
            "LORA_RANK": "0",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "1.05",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "2.20",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-0.70",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.85",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.004",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.035",
            "XY_LOSS_WEIGHT": "0.16",
            "SPEED_LOSS_WEIGHT": "0.24",
            "MOVING_SAMPLE_WEIGHT": "1.20",
            "STOPPED_SAMPLE_WEIGHT": "2.20",
            "HAZARD_SAMPLE_WEIGHT": "6.50",
            "STOPSIGN_SAMPLE_WEIGHT": "3.00",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "2.20",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "2.20",
            "STOP_LOSS_WEIGHT": "0.16",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.18",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.16",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "1.45",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.00",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.55",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.35",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.20",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.070",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.120",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.120",
            "CONTROL_LOSS_WEIGHT": "0.85",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.80",
            "PDM_LATERAL_LOSS_WEIGHT": "0.35",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.35",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.90",
            "PDM_CONTROLLER_LOSS_WEIGHT": "1.45",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.55",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.55",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "1.45",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "4.80",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "4.20",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "2.80",
            "CONTROLLER_GO_HINGE_MARGIN": "0.05",
            "CONTROLLER_SOFT_STOP_TEMPERATURE": "0.80",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.05",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.15",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "7.2",
        },
    ),
    Recipe(
        "hinge_stop_go_balance_b24",
        {
            "BATCH_SIZE": "24",
            "LR": "3.4e-5",
            "WEIGHT_DECAY": "9e-5",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16.0",
            "LORA_DROPOUT": "0.02",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.90",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "1.90",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-0.85",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.75",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.006",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.050",
            "XY_LOSS_WEIGHT": "0.24",
            "SPEED_LOSS_WEIGHT": "0.34",
            "MOVING_SAMPLE_WEIGHT": "1.10",
            "STOPPED_SAMPLE_WEIGHT": "2.50",
            "HAZARD_SAMPLE_WEIGHT": "7.00",
            "TRAFFICLIGHT_SAMPLE_WEIGHT": "1.60",
            "STOPSIGN_SAMPLE_WEIGHT": "3.20",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "2.30",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "2.20",
            "STOP_LOSS_WEIGHT": "0.17",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.18",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.16",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "1.55",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.05",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.70",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.45",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.35",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.075",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.130",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.130",
            "CONTROL_LOSS_WEIGHT": "0.95",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.90",
            "PDM_LATERAL_LOSS_WEIGHT": "0.55",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.45",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "1.00",
            "PDM_CONTROLLER_LOSS_WEIGHT": "1.65",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.85",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.65",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "1.55",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "4.50",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "3.20",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "3.20",
            "CONTROLLER_GO_HINGE_MARGIN": "0.05",
            "CONTROLLER_SOFT_STOP_TEMPERATURE": "0.65",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.20",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.25",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "7.8",
        },
    ),
]

SECONDGEN_RECIPES = [
    Recipe(
        "sg_closed_progress_lora_b16",
        {
            "BATCH_SIZE": "16",
            "LR": "2.2e-5",
            "WEIGHT_DECAY": "1e-4",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "16",
            "LORA_ALPHA": "32.0",
            "LORA_DROPOUT": "0.03",
            "UNFREEZE_INCLUDE": "^join\\.layers\\.(4|5)\\.",
            "UNFREEZE_LR": "2.0e-6",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.65",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "1.35",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-1.25",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.55",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.012",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.090",
            "XY_LOSS_WEIGHT": "0.42",
            "SPEED_LOSS_WEIGHT": "0.40",
            "MOVING_SAMPLE_WEIGHT": "1.10",
            "STOPPED_SAMPLE_WEIGHT": "1.80",
            "HAZARD_SAMPLE_WEIGHT": "4.50",
            "STOPSIGN_SAMPLE_WEIGHT": "2.00",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "1.60",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "1.60",
            "STOP_LOSS_WEIGHT": "0.10",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.12",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.10",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "0.90",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.060",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.120",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.120",
            "CONTROL_LOSS_WEIGHT": "0.65",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.55",
            "PDM_LATERAL_LOSS_WEIGHT": "1.20",
            "PDM_PROGRESS_LOSS_WEIGHT": "1.20",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.55",
            "PDM_CONTROLLER_LOSS_WEIGHT": "1.10",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "1.20",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.55",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "0.95",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "3.20",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "2.20",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "5.50",
            "CONTROLLER_GO_HINGE_MARGIN": "0.04",
            "CONTROLLER_SOFT_STOP_TEMPERATURE": "0.80",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.80",
            "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.25",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.30",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "8.0",
        },
    ),
    Recipe(
        "sg_small_residual_anchor_b24",
        {
            "BATCH_SIZE": "24",
            "LR": "2.8e-5",
            "WEIGHT_DECAY": "8e-5",
            "FREEZE_TASK_ADAPTER": "1",
            "LORA_RANK": "0",
            "INIT_PARAM_ANCHOR_LOSS_WEIGHT": "0.0010",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.45",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "1.10",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-1.80",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.35",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.020",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.160",
            "XY_LOSS_WEIGHT": "0.50",
            "SPEED_LOSS_WEIGHT": "0.45",
            "MOVING_SAMPLE_WEIGHT": "1.00",
            "STOPPED_SAMPLE_WEIGHT": "1.70",
            "HAZARD_SAMPLE_WEIGHT": "4.00",
            "STOPSIGN_SAMPLE_WEIGHT": "1.80",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "1.50",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "1.50",
            "STOP_LOSS_WEIGHT": "0.10",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.10",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.10",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "0.85",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.050",
            "CONTROL_LOSS_WEIGHT": "0.55",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.45",
            "PDM_LATERAL_LOSS_WEIGHT": "1.00",
            "PDM_PROGRESS_LOSS_WEIGHT": "1.00",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.50",
            "PDM_CONTROLLER_LOSS_WEIGHT": "0.90",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.90",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.45",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "0.85",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "2.80",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "2.00",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "4.80",
            "CONTROLLER_GO_HINGE_MARGIN": "0.04",
            "CONTROLLER_SOFT_STOP_TEMPERATURE": "0.90",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.60",
            "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.30",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.25",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "8.0",
        },
    ),
    Recipe(
        "sg_speed_head_go_b16",
        {
            "BATCH_SIZE": "16",
            "LR": "1.6e-5",
            "WEIGHT_DECAY": "1e-4",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16.0",
            "LORA_DROPOUT": "0.02",
            "UNFREEZE_INCLUDE": "^target_speed_network\\.,^checkpoint_decoder\\.(encoder|decoder)\\.",
            "UNFREEZE_LR": "4.0e-6",
            "UNFREEZE_WEIGHT_DECAY": "5e-6",
            "INIT_PARAM_ANCHOR_LOSS_WEIGHT": "0.0005",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.75",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "1.60",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-1.15",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.60",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.010",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.080",
            "XY_LOSS_WEIGHT": "0.34",
            "SPEED_LOSS_WEIGHT": "0.52",
            "MOVING_SAMPLE_WEIGHT": "1.30",
            "STOPPED_SAMPLE_WEIGHT": "1.60",
            "HAZARD_SAMPLE_WEIGHT": "4.20",
            "STOP_LOSS_WEIGHT": "0.09",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.10",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.08",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "0.80",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.120",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.220",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.200",
            "CONTROL_LOSS_WEIGHT": "0.70",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.55",
            "PDM_LATERAL_LOSS_WEIGHT": "0.80",
            "PDM_PROGRESS_LOSS_WEIGHT": "1.10",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.55",
            "PDM_CONTROLLER_LOSS_WEIGHT": "1.00",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.80",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.75",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "0.85",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "4.80",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "2.60",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "5.20",
            "CONTROLLER_GO_HINGE_MARGIN": "0.07",
            "CONTROLLER_SOFT_STOP_TEMPERATURE": "0.85",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.45",
            "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.20",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.20",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "8.5",
        },
    ),
    Recipe(
        "sg_low_stop_path_b24",
        {
            "BATCH_SIZE": "24",
            "LR": "3.2e-5",
            "WEIGHT_DECAY": "8e-5",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16.0",
            "LORA_DROPOUT": "0.02",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.70",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "1.40",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-1.30",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.55",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.012",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.090",
            "XY_LOSS_WEIGHT": "0.55",
            "SPEED_LOSS_WEIGHT": "0.38",
            "MOVING_SAMPLE_WEIGHT": "1.10",
            "STOPPED_SAMPLE_WEIGHT": "1.50",
            "HAZARD_SAMPLE_WEIGHT": "3.50",
            "STOP_LOSS_WEIGHT": "0.07",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.08",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.08",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "0.65",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.060",
            "CONTROL_LOSS_WEIGHT": "0.55",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.40",
            "PDM_LATERAL_LOSS_WEIGHT": "1.40",
            "PDM_PROGRESS_LOSS_WEIGHT": "1.40",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.35",
            "PDM_CONTROLLER_LOSS_WEIGHT": "0.85",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "1.30",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.45",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "0.70",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "2.80",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "1.80",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "4.80",
            "CONTROLLER_GO_HINGE_MARGIN": "0.04",
            "CONTROLLER_SOFT_STOP_TEMPERATURE": "0.90",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "2.00",
            "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.35",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.35",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "8.0",
        },
    ),
]

DIRECT_CONTROLLER_RECIPES = [
    Recipe(
        "direct_action_residual_b16",
        {
            "BATCH_SIZE": "16",
            "LR": "4.0e-5",
            "WEIGHT_DECAY": "8e-5",
            "FREEZE_TASK_ADAPTER": "1",
            "LORA_RANK": "0",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.45",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "0.90",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-1.20",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.50",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.001",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.002",
            "OUTPUT_RESIDUAL_CONTROLLER": "1",
            "OUTPUT_RESIDUAL_CONTROLLER_LATERAL_SCALE": "2.20",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "2.40",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "5.50",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_BIAS": "-0.45",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_MAX": "1.25",
            "DIRECT_CONTROL_LOSS_WEIGHT": "6.00",
            "XY_LOSS_WEIGHT": "0.22",
            "SPEED_LOSS_WEIGHT": "0.22",
            "MOVING_SAMPLE_WEIGHT": "1.15",
            "STOPPED_SAMPLE_WEIGHT": "2.10",
            "HAZARD_SAMPLE_WEIGHT": "5.50",
            "STOPSIGN_SAMPLE_WEIGHT": "2.40",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "2.20",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "2.00",
            "STOP_LOSS_WEIGHT": "0.12",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.10",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.08",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "1.00",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.080",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.180",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.160",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.70",
            "PDM_LATERAL_LOSS_WEIGHT": "1.10",
            "PDM_PROGRESS_LOSS_WEIGHT": "1.10",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.80",
            "PDM_CONTROLLER_LOSS_WEIGHT": "4.50",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "2.20",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "2.50",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "3.50",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "5.50",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "3.50",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "4.50",
            "CONTROLLER_GO_HINGE_MARGIN": "0.06",
            "CONTROLLER_SOFT_STOP_TEMPERATURE": "0.70",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "2.20",
            "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.20",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.35",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "8.5",
        },
    ),
    Recipe(
        "direct_lora_controller_path_b16",
        {
            "BATCH_SIZE": "16",
            "LR": "2.4e-5",
            "WEIGHT_DECAY": "1e-4",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "16",
            "LORA_ALPHA": "32.0",
            "LORA_DROPOUT": "0.03",
            "UNFREEZE_INCLUDE": "^join\\.layers\\.(4|5)\\.",
            "UNFREEZE_LR": "2.0e-6",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.35",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "0.75",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-1.35",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.45",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.002",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.004",
            "OUTPUT_RESIDUAL_CONTROLLER": "1",
            "OUTPUT_RESIDUAL_CONTROLLER_LATERAL_SCALE": "1.80",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "2.20",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "5.00",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_BIAS": "-0.65",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_MAX": "1.10",
            "DIRECT_CONTROL_LOSS_WEIGHT": "5.00",
            "XY_LOSS_WEIGHT": "0.34",
            "SPEED_LOSS_WEIGHT": "0.28",
            "MOVING_SAMPLE_WEIGHT": "1.20",
            "STOPPED_SAMPLE_WEIGHT": "1.90",
            "HAZARD_SAMPLE_WEIGHT": "4.80",
            "STOPSIGN_SAMPLE_WEIGHT": "2.20",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "1.90",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "1.80",
            "STOP_LOSS_WEIGHT": "0.10",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.10",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.08",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "0.90",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.100",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.220",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.200",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.60",
            "PDM_LATERAL_LOSS_WEIGHT": "1.40",
            "PDM_PROGRESS_LOSS_WEIGHT": "1.30",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.65",
            "PDM_CONTROLLER_LOSS_WEIGHT": "3.80",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "2.60",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "2.20",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "2.80",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "6.00",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "3.20",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "5.50",
            "CONTROLLER_GO_HINGE_MARGIN": "0.07",
            "CONTROLLER_SOFT_STOP_TEMPERATURE": "0.75",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "2.40",
            "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.30",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.35",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "8.5",
        },
    ),
    Recipe(
        "direct_brake_gate_b24",
        {
            "BATCH_SIZE": "24",
            "LR": "3.5e-5",
            "WEIGHT_DECAY": "8e-5",
            "FREEZE_TASK_ADAPTER": "1",
            "LORA_RANK": "0",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.30",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "0.60",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-1.50",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.35",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.002",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.003",
            "OUTPUT_RESIDUAL_CONTROLLER": "1",
            "OUTPUT_RESIDUAL_CONTROLLER_LATERAL_SCALE": "1.20",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "1.40",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "7.00",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_BIAS": "-0.40",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_MAX": "1.30",
            "DIRECT_CONTROL_LOSS_WEIGHT": "7.50",
            "XY_LOSS_WEIGHT": "0.24",
            "SPEED_LOSS_WEIGHT": "0.20",
            "MOVING_SAMPLE_WEIGHT": "0.95",
            "STOPPED_SAMPLE_WEIGHT": "2.80",
            "HAZARD_SAMPLE_WEIGHT": "8.00",
            "TRAFFICLIGHT_SAMPLE_WEIGHT": "1.80",
            "STOPSIGN_SAMPLE_WEIGHT": "3.80",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "3.20",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "3.00",
            "STOP_LOSS_WEIGHT": "0.18",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.16",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.14",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "1.60",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.10",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.60",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.40",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.30",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.040",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.80",
            "PDM_LATERAL_LOSS_WEIGHT": "0.75",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.60",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "1.20",
            "PDM_CONTROLLER_LOSS_WEIGHT": "4.20",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "1.40",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "1.60",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "4.50",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "3.00",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "3.20",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "2.80",
            "CONTROLLER_GO_HINGE_MARGIN": "0.04",
            "CONTROLLER_SOFT_STOP_TEMPERATURE": "0.60",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.80",
            "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.15",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.20",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "7.0",
        },
    ),
    Recipe(
        "direct_go_progress_b16",
        {
            "BATCH_SIZE": "16",
            "LR": "3.0e-5",
            "WEIGHT_DECAY": "9e-5",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16.0",
            "LORA_DROPOUT": "0.02",
            "UNFREEZE_INCLUDE": "^target_speed_network\\.,^checkpoint_decoder\\.(encoder|decoder)\\.",
            "UNFREEZE_LR": "4.0e-6",
            "UNFREEZE_WEIGHT_DECAY": "5e-6",
            "INIT_PARAM_ANCHOR_LOSS_WEIGHT": "0.0004",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.35",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "0.85",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-1.35",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.45",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.002",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.004",
            "OUTPUT_RESIDUAL_CONTROLLER": "1",
            "OUTPUT_RESIDUAL_CONTROLLER_LATERAL_SCALE": "1.60",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "3.00",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "6.20",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_BIAS": "-0.55",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_MAX": "1.20",
            "DIRECT_CONTROL_LOSS_WEIGHT": "5.50",
            "XY_LOSS_WEIGHT": "0.30",
            "SPEED_LOSS_WEIGHT": "0.36",
            "MOVING_SAMPLE_WEIGHT": "1.45",
            "STOPPED_SAMPLE_WEIGHT": "1.70",
            "HAZARD_SAMPLE_WEIGHT": "4.40",
            "STOPSIGN_SAMPLE_WEIGHT": "2.10",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "1.70",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "1.70",
            "STOP_LOSS_WEIGHT": "0.10",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.10",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.08",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "0.85",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.140",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.260",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.240",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.60",
            "PDM_LATERAL_LOSS_WEIGHT": "1.10",
            "PDM_PROGRESS_LOSS_WEIGHT": "1.60",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.55",
            "PDM_CONTROLLER_LOSS_WEIGHT": "3.60",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "2.00",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "2.80",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "2.40",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "7.00",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "3.40",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "7.50",
            "CONTROLLER_GO_HINGE_MARGIN": "0.08",
            "CONTROLLER_SOFT_STOP_TEMPERATURE": "0.80",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "2.60",
            "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.35",
            "CLOSED_LOOP_UNROLL_STEER_GAIN": "1.35",
            "CLOSED_LOOP_UNROLL_MAX_SPEED_MPS": "9.0",
        },
    ),
]

METRIC_DIRECT_RECIPES = [
    Recipe(
        "metric_balanced_direct_b16",
        {
            **DIRECT_CONTROLLER_RECIPES[0].env,
            "LR": "3.2e-5",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "1.80",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "4.20",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_BIAS": "-0.65",
            "DIRECT_CONTROL_LOSS_WEIGHT": "4.00",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "3.50",
            "SAFE_CTRL_SURROGATE_TEMPERATURE": "0.22",
            "PDM_CONTROLLER_LOSS_WEIGHT": "2.40",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "1.70",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "1.60",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "1.80",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "4.50",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "3.50",
            "MOVING_SAMPLE_WEIGHT": "1.25",
            "STOPPED_SAMPLE_WEIGHT": "2.00",
            "HAZARD_SAMPLE_WEIGHT": "5.20",
        },
    ),
    Recipe(
        "metric_go_hazard_lora_b16",
        {
            **DIRECT_CONTROLLER_RECIPES[3].env,
            "LR": "2.2e-5",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "2.30",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "4.80",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_BIAS": "-0.75",
            "DIRECT_CONTROL_LOSS_WEIGHT": "4.50",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "4.20",
            "SAFE_CTRL_SURROGATE_TEMPERATURE": "0.20",
            "PDM_CONTROLLER_LOSS_WEIGHT": "2.80",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "1.80",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "1.90",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "2.20",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "5.50",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "5.50",
            "HAZARD_SAMPLE_WEIGHT": "5.80",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "2.30",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "2.20",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "2.20",
        },
    ),
    Recipe(
        "metric_small_residual_b24",
        {
            **DIRECT_CONTROLLER_RECIPES[0].env,
            "BATCH_SIZE": "24",
            "LR": "2.8e-5",
            "FREEZE_TASK_ADAPTER": "1",
            "LORA_RANK": "0",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.25",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "0.55",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-1.55",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.35",
            "OUTPUT_RESIDUAL_CONTROLLER_LATERAL_SCALE": "1.40",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "1.60",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "4.50",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_BIAS": "-0.85",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_MAX": "1.00",
            "DIRECT_CONTROL_LOSS_WEIGHT": "3.20",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "5.00",
            "SAFE_CTRL_SURROGATE_TEMPERATURE": "0.18",
            "XY_LOSS_WEIGHT": "0.40",
            "SPEED_LOSS_WEIGHT": "0.32",
            "PDM_CONTROLLER_LOSS_WEIGHT": "2.20",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "1.60",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "1.40",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "1.70",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.60",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.010",
        },
    ),
]

FASTLR_RECIPES = [
    Recipe(
        "fast_go_hazard_lora_lr4e5_b16",
        {
            **METRIC_DIRECT_RECIPES[1].env,
            "LR": "4.0e-5",
            "UNFREEZE_LR": "8.0e-6",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "5.20",
            "DIRECT_CONTROL_LOSS_WEIGHT": "5.20",
            "PDM_CONTROLLER_LOSS_WEIGHT": "3.20",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "6.50",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "6.20",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "2.60",
        },
    ),
    Recipe(
        "fast_go_hazard_lora_lr6e5_b16",
        {
            **METRIC_DIRECT_RECIPES[1].env,
            "LR": "6.0e-5",
            "WEIGHT_DECAY": "7e-5",
            "UNFREEZE_LR": "1.0e-5",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "6.00",
            "DIRECT_CONTROL_LOSS_WEIGHT": "5.80",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "2.80",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "5.60",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "7.00",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "7.20",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "2.90",
        },
    ),
    Recipe(
        "fast_direct_action_lr6e5_b16",
        {
            **METRIC_DIRECT_RECIPES[0].env,
            "LR": "6.0e-5",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "5.50",
            "DIRECT_CONTROL_LOSS_WEIGHT": "5.20",
            "OUTPUT_RESIDUAL_CONTROLLER_LATERAL_SCALE": "2.60",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "2.70",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "6.00",
            "PDM_CONTROLLER_LOSS_WEIGHT": "3.20",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "2.10",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "2.40",
        },
    ),
    Recipe(
        "fast_direct_action_lr9e5_b16",
        {
            **METRIC_DIRECT_RECIPES[0].env,
            "LR": "9.0e-5",
            "WEIGHT_DECAY": "6e-5",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "6.20",
            "DIRECT_CONTROL_LOSS_WEIGHT": "6.00",
            "OUTPUT_RESIDUAL_CONTROLLER_LATERAL_SCALE": "2.80",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "3.10",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "6.80",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_MAX": "1.35",
            "PDM_CONTROLLER_LOSS_WEIGHT": "3.60",
        },
    ),
    Recipe(
        "fast_small_residual_lr8e5_b24",
        {
            **METRIC_DIRECT_RECIPES[2].env,
            "LR": "8.0e-5",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "6.50",
            "DIRECT_CONTROL_LOSS_WEIGHT": "4.20",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "2.20",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "5.50",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_MAX": "1.20",
            "PDM_CONTROLLER_LOSS_WEIGHT": "2.80",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "2.20",
        },
    ),
    Recipe(
        "fast_speed_decoder_lr5e5_b16",
        {
            **METRIC_DIRECT_RECIPES[1].env,
            "LR": "5.0e-5",
            "UNFREEZE_INCLUDE": "^target_speed_network\\.,^checkpoint_decoder\\.(encoder|decoder)\\.",
            "UNFREEZE_LR": "1.2e-5",
            "UNFREEZE_WEIGHT_DECAY": "3e-6",
            "INIT_PARAM_ANCHOR_LOSS_WEIGHT": "0.0002",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "5.80",
            "DIRECT_CONTROL_LOSS_WEIGHT": "5.40",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "6.20",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.220",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.340",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.320",
        },
    ),
    Recipe(
        "fast_brake_gate_lr7e5_b24",
        {
            **DIRECT_CONTROLLER_RECIPES[2].env,
            "LR": "7.0e-5",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "5.80",
            "DIRECT_CONTROL_LOSS_WEIGHT": "8.00",
            "PDM_CONTROLLER_LOSS_WEIGHT": "4.80",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "5.20",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "7.80",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_MAX": "1.35",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "2.20",
        },
    ),
    Recipe(
        "fast_go_progress_lr7e5_b16",
        {
            **DIRECT_CONTROLLER_RECIPES[3].env,
            "LR": "7.0e-5",
            "UNFREEZE_LR": "1.0e-5",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "5.80",
            "DIRECT_CONTROL_LOSS_WEIGHT": "6.20",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "3.60",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "6.80",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "8.50",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "8.00",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "3.20",
        },
    ),
]

FASTLR_E1_RESUME_CHECKPOINT = "/home/jovyan/dataset/byeongjae/runs/benchmix8_safe_ctrl_fastlr_queue_target_only/fast_go_hazard_lora_lr4e5_b16/epoch_checkpoints/epoch_001.pt"

FASTLR_RESUME_RECIPES = [
    Recipe(
        "fast_go_hazard_lora_lr4e5_b16_resume_e1",
        {
            **FASTLR_RECIPES[0].env,
            "INIT_CHECKPOINT": FASTLR_E1_RESUME_CHECKPOINT,
        },
    ),
    *FASTLR_RECIPES[2:],
]

CONSERVATIVE_RECIPES = [
    Recipe(
        "conservative_residual_progress_lora_b16",
        {
            "BATCH_SIZE": "16",
            "LR": "2.5e-5",
            "WEIGHT_DECAY": "1e-4",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16.0",
            "LORA_DROPOUT": "0.02",
            "UNFREEZE_INCLUDE": "",
            "OUTPUT_RESIDUAL_CONTROLLER": "0",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.75",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "1.60",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-1.20",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.65",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.010",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.075",
            "FEATURE_DRIFT_LOSS_WEIGHT": "0.16",
            "OUTPUT_PRIOR_XY_LOSS_WEIGHT": "0.05",
            "OUTPUT_PRIOR_SPEED_LOSS_WEIGHT": "0.10",
            "XY_LOSS_WEIGHT": "0.24",
            "SPEED_LOSS_WEIGHT": "0.35",
            "MOVING_SAMPLE_WEIGHT": "1.25",
            "STOPPED_SAMPLE_WEIGHT": "2.10",
            "HAZARD_SAMPLE_WEIGHT": "7.00",
            "STOPSIGN_SAMPLE_WEIGHT": "3.00",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "2.50",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "2.20",
            "STOP_LOSS_WEIGHT": "0.16",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.18",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.16",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "1.40",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "0.90",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.45",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.35",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.20",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.090",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.160",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.160",
            "CONTROL_LOSS_WEIGHT": "0.90",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.80",
            "PDM_LATERAL_LOSS_WEIGHT": "0.40",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.55",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.90",
            "PDM_CONTROLLER_LOSS_WEIGHT": "1.50",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.70",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.65",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "1.30",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "4.80",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "2.80",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "4.80",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.30",
            "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.10",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "2.80",
            "SAFE_CTRL_SURROGATE_TEMPERATURE": "0.28",
        },
    ),
    Recipe(
        "conservative_exclusive_direct_b16",
        {
            "BATCH_SIZE": "16",
            "LR": "2.2e-5",
            "WEIGHT_DECAY": "1e-4",
            "FREEZE_TASK_ADAPTER": "0",
            "LORA_RANK": "8",
            "LORA_ALPHA": "16.0",
            "LORA_DROPOUT": "0.02",
            "OUTPUT_RESIDUAL_CONTROLLER": "1",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.65",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "1.40",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-1.35",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.60",
            "OUTPUT_RESIDUAL_CONTROLLER_LATERAL_SCALE": "0.90",
            "OUTPUT_RESIDUAL_CONTROLLER_PROGRESS_SCALE": "1.20",
            "OUTPUT_RESIDUAL_CONTROLLER_SPEED_LOGIT_SCALE": "2.00",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_BIAS": "-1.70",
            "OUTPUT_RESIDUAL_CONTROLLER_GATE_MAX": "0.55",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.020",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.090",
            "DIRECT_CONTROL_LOSS_WEIGHT": "1.80",
            "DIRECT_CONTROL_COACTIVATION_LOSS_WEIGHT": "10.00",
            "FEATURE_DRIFT_LOSS_WEIGHT": "0.18",
            "OUTPUT_PRIOR_XY_LOSS_WEIGHT": "0.05",
            "OUTPUT_PRIOR_SPEED_LOSS_WEIGHT": "0.12",
            "XY_LOSS_WEIGHT": "0.24",
            "SPEED_LOSS_WEIGHT": "0.36",
            "MOVING_SAMPLE_WEIGHT": "1.20",
            "STOPPED_SAMPLE_WEIGHT": "2.10",
            "HAZARD_SAMPLE_WEIGHT": "6.50",
            "STOPSIGN_SAMPLE_WEIGHT": "3.00",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "2.50",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "2.20",
            "STOP_LOSS_WEIGHT": "0.16",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.18",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.16",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "1.30",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "0.85",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.35",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.30",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.15",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.080",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.140",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.140",
            "CONTROL_LOSS_WEIGHT": "0.85",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.75",
            "PDM_LATERAL_LOSS_WEIGHT": "0.40",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.50",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.85",
            "PDM_CONTROLLER_LOSS_WEIGHT": "1.45",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.70",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.65",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "1.25",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "4.20",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "2.60",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "4.20",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.20",
            "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.10",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "2.60",
            "SAFE_CTRL_SURROGATE_TEMPERATURE": "0.30",
        },
    ),
    Recipe(
        "conservative_stop_go_balance_b24",
        {
            "BATCH_SIZE": "24",
            "LR": "3.0e-5",
            "WEIGHT_DECAY": "9e-5",
            "FREEZE_TASK_ADAPTER": "1",
            "LORA_RANK": "0",
            "OUTPUT_RESIDUAL_CONTROLLER": "0",
            "OUTPUT_RESIDUAL_CHECKPOINT_SCALE": "0.70",
            "OUTPUT_RESIDUAL_SPEED_LOGIT_SCALE": "1.55",
            "OUTPUT_RESIDUAL_GATE_BIAS": "-1.25",
            "OUTPUT_RESIDUAL_GATE_MAX": "0.60",
            "OUTPUT_RESIDUAL_GATE_LOSS_WEIGHT": "0.012",
            "OUTPUT_RESIDUAL_NORM_LOSS_WEIGHT": "0.080",
            "FEATURE_DRIFT_LOSS_WEIGHT": "0.18",
            "OUTPUT_PRIOR_XY_LOSS_WEIGHT": "0.06",
            "OUTPUT_PRIOR_SPEED_LOSS_WEIGHT": "0.12",
            "XY_LOSS_WEIGHT": "0.22",
            "SPEED_LOSS_WEIGHT": "0.34",
            "MOVING_SAMPLE_WEIGHT": "1.15",
            "STOPPED_SAMPLE_WEIGHT": "2.35",
            "HAZARD_SAMPLE_WEIGHT": "7.50",
            "TRAFFICLIGHT_SAMPLE_WEIGHT": "1.35",
            "STOPSIGN_SAMPLE_WEIGHT": "3.40",
            "FRONTVEHICLE_SAMPLE_WEIGHT": "2.70",
            "JUNCTIONYIELD_SAMPLE_WEIGHT": "2.35",
            "STOP_LOSS_WEIGHT": "0.17",
            "STOP_STATE_AUX_LOSS_WEIGHT": "0.18",
            "STOP_REASON_AUX_LOSS_WEIGHT": "0.16",
            "STOP_SPEED_CEILING_LOSS_WEIGHT": "1.55",
            "TRAFFICLIGHT_STOP_SPEED_CEILING_LOSS_WEIGHT": "0.95",
            "STOPSIGN_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.65",
            "FRONTVEHICLE_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.45",
            "JUNCTIONYIELD_STOP_SPEED_CEILING_LOSS_WEIGHT": "1.30",
            "SPEED_FLOOR_LOSS_WEIGHT": "0.075",
            "LAUNCH_SPEED_FLOOR_LOSS_WEIGHT": "0.130",
            "RELEASE_SPEED_FLOOR_LOSS_WEIGHT": "0.130",
            "CONTROL_LOSS_WEIGHT": "0.85",
            "PDM_BEHAVIOR_LOSS_WEIGHT": "0.80",
            "PDM_LATERAL_LOSS_WEIGHT": "0.35",
            "PDM_PROGRESS_LOSS_WEIGHT": "0.45",
            "PDM_HAZARD_PROGRESS_LOSS_WEIGHT": "0.95",
            "PDM_CONTROLLER_LOSS_WEIGHT": "1.45",
            "PDM_PLAN_STEER_LOSS_WEIGHT": "0.55",
            "PDM_PLAN_THROTTLE_LOSS_WEIGHT": "0.55",
            "PDM_PLAN_BRAKE_LOSS_WEIGHT": "1.45",
            "CONTROLLER_GO_HINGE_LOSS_WEIGHT": "4.00",
            "CONTROLLER_THROTTLE_CLOSE_HINGE_LOSS_WEIGHT": "2.80",
            "PDM_GO_PROGRESS_HINGE_LOSS_WEIGHT": "3.80",
            "CLOSED_LOOP_UNROLL_LOSS_WEIGHT": "1.10",
            "CLOSED_LOOP_UNROLL_TARGET_SPEED_BLEND": "0.10",
            "SAFE_CTRL_SURROGATE_LOSS_WEIGHT": "2.70",
            "SAFE_CTRL_SURROGATE_TEMPERATURE": "0.30",
        },
    ),
]

if RECIPE_SET in {"conservative", "safe_conservative", "exclusive"}:
    RECIPES = CONSERVATIVE_RECIPES
elif RECIPE_SET in {"fastlr_resume", "fast_lr_resume", "metric_fastlr_resume"}:
    RECIPES = FASTLR_RESUME_RECIPES
elif RECIPE_SET in {"fastlr", "fast_lr", "metric_fastlr", "metric_lr_sweep"}:
    RECIPES = FASTLR_RECIPES
elif RECIPE_SET in {"metric", "direct_metric", "soft_metric"}:
    RECIPES = METRIC_DIRECT_RECIPES
elif RECIPE_SET in {"direct", "controller", "controller_residual"}:
    RECIPES = DIRECT_CONTROLLER_RECIPES
elif RECIPE_SET in {"hinge2", "secondgen"}:
    RECIPES = SECONDGEN_RECIPES
elif RECIPE_SET == "hinge":
    RECIPES = HINGE_RECIPES
else:
    RECIPES = BASE_RECIPES

if ONLY_RECIPES:
    RECIPES = [recipe for recipe in RECIPES if recipe.name in ONLY_RECIPES]
if SKIP_RECIPES:
    RECIPES = [recipe for recipe in RECIPES if recipe.name not in SKIP_RECIPES]


@dataclass
class Active:
    recipe: Recipe
    gpu: int
    session: str
    log_path: pathlib.Path
    out_dir: pathlib.Path
    epoch1_checked: bool = False
    epoch2_checked: bool = False
    promoted: bool = False


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def tmux_has(session: str) -> bool:
    return run(["tmux", "has-session", "-t", session]).returncode == 0


def tmux_kill(session: str) -> None:
    run(["tmux", "kill-session", "-t", session])


def gpu_mem_mb() -> dict[int, int]:
    proc = run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"])
    result: dict[int, int] = {}
    if proc.returncode != 0:
        return result
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                result[int(parts[0])] = int(parts[1])
            except ValueError:
                pass
    return result


def parse_epoch_safe_ctrl(log_path: pathlib.Path, epoch: int) -> float | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    epoch_text = f"{epoch:03d}"
    matches = re.findall(rf"epoch={epoch_text}[^\n]*safe_ctrl=([0-9]+(?:\.[0-9]+)?)", text)
    if matches:
        return float(matches[-1])
    matches = re.findall(rf"epoch={epoch_text}[^\n]*select=safety_controller_closed_loop_proxy:([0-9]+(?:\.[0-9]+)?)", text)
    if matches:
        return float(matches[-1])
    return None


def launch(recipe: Recipe, gpu: int) -> Active:
    out_dir = RUN_ROOT / recipe.name
    log_path = LOG_DIR / f"{recipe.name}.log"
    session = f"t2d_safeq_{recipe.name[:36]}_g{gpu}"
    env = dict(COMMON)
    env.update(recipe.env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["WORK_ROOT"] = str(RUN_ROOT / f"{recipe.name}_work")
    env["OUT"] = str(out_dir)
    env["EPOCHS"] = str(PROMOTE_EPOCHS)

    command = "env " + " ".join(f"{shlex.quote(k + '=' + v)}" for k, v in env.items())
    command += " bash configs/train_tfpp_tesla_town13_task_feature_adapter_server.sh"
    command += f" > {shlex.quote(str(log_path))} 2>&1"

    if tmux_has(session):
        raise RuntimeError(f"session already exists: {session}")
    run(["tmux", "new-session", "-d", "-s", session, "-c", str(ADAPTER_ROOT), "--", "bash", "-lc", command], check=True)
    return Active(recipe=recipe, gpu=gpu, session=session, log_path=log_path, out_dir=out_dir)


def append_decision(event: dict[str, object]) -> None:
    status_path = LOG_DIR / "safe_ctrl_queue_decisions.jsonl"
    with status_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def main() -> int:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not TARGET_VIEW.is_dir() or not TARGET_INDEX.is_file():
        raise SystemExit(f"missing prepared target view/index: {TARGET_VIEW} {TARGET_INDEX}")
    if not INIT_CHECKPOINT.is_file():
        raise SystemExit(f"missing init checkpoint: {INIT_CHECKPOINT}")

    queue = list(RECIPES)
    active: dict[int, Active] = {}
    print(
        f"safe_ctrl queue start epoch1_threshold={EPOCH1_THRESHOLD} "
        f"epoch2_threshold={EPOCH2_THRESHOLD} epochs={PROMOTE_EPOCHS} "
        f"baseline_safe_ctrl={BASELINE_SAFE_CTRL} gpus={GPU_IDS} "
        f"recipe_set={RECIPE_SET} recipes={len(queue)}",
        flush=True,
    )
    print(f"logs={LOG_DIR}", flush=True)

    while queue or active:
        for gpu, job in list(active.items()):
            epoch1_safe_ctrl = parse_epoch_safe_ctrl(job.log_path, 1)
            if epoch1_safe_ctrl is not None and not job.epoch1_checked:
                job.epoch1_checked = True
                event = {
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "recipe": job.recipe.name,
                    "gpu": gpu,
                    "session": job.session,
                    "log": str(job.log_path),
                    "out": str(job.out_dir),
                    "epoch1_safe_ctrl": epoch1_safe_ctrl,
                    "epoch1_threshold": EPOCH1_THRESHOLD,
                    "epoch2_threshold": EPOCH2_THRESHOLD,
                }
                if epoch1_safe_ctrl <= EPOCH1_THRESHOLD:
                    event["decision"] = "epoch1_continue"
                    append_decision(event)
                    print(
                        f"EPOCH1_CONTINUE gpu={gpu} recipe={job.recipe.name} "
                        f"safe_ctrl={epoch1_safe_ctrl:.6f}",
                        flush=True,
                    )
                else:
                    event["decision"] = "reject_kill"
                    append_decision(event)
                    print(
                        f"REJECT_EPOCH1 gpu={gpu} recipe={job.recipe.name} "
                        f"safe_ctrl={epoch1_safe_ctrl:.6f}; killing",
                        flush=True,
                    )
                    tmux_kill(job.session)
                    del active[gpu]
                continue

            epoch2_safe_ctrl = parse_epoch_safe_ctrl(job.log_path, 2)
            if epoch2_safe_ctrl is not None and not job.epoch2_checked:
                job.epoch2_checked = True
                event = {
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "recipe": job.recipe.name,
                    "gpu": gpu,
                    "session": job.session,
                    "log": str(job.log_path),
                    "out": str(job.out_dir),
                    "epoch1_safe_ctrl": epoch1_safe_ctrl,
                    "epoch2_safe_ctrl": epoch2_safe_ctrl,
                    "epoch1_threshold": EPOCH1_THRESHOLD,
                    "epoch2_threshold": EPOCH2_THRESHOLD,
                }
                if epoch2_safe_ctrl <= EPOCH2_THRESHOLD:
                    job.promoted = True
                    event["decision"] = "epoch2_promote_full"
                    append_decision(event)
                    print(
                        f"PROMOTE_EPOCH2 gpu={gpu} recipe={job.recipe.name} "
                        f"safe_ctrl={epoch2_safe_ctrl:.6f}; continuing full run",
                        flush=True,
                    )
                else:
                    event["decision"] = "reject_kill_epoch2"
                    append_decision(event)
                    print(
                        f"REJECT_EPOCH2 gpu={gpu} recipe={job.recipe.name} "
                        f"safe_ctrl={epoch2_safe_ctrl:.6f}; killing",
                        flush=True,
                    )
                    tmux_kill(job.session)
                    del active[gpu]
                continue

            if not tmux_has(job.session):
                event = {
                    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "recipe": job.recipe.name,
                    "gpu": gpu,
                    "session": job.session,
                    "log": str(job.log_path),
                    "out": str(job.out_dir),
                    "epoch1_safe_ctrl": epoch1_safe_ctrl,
                    "epoch2_safe_ctrl": epoch2_safe_ctrl,
                    "decision": "completed" if job.epoch1_checked else "ended_without_epoch1",
                    "epoch1_threshold": EPOCH1_THRESHOLD,
                    "epoch2_threshold": EPOCH2_THRESHOLD,
                    "promoted": job.promoted,
                }
                append_decision(event)
                if job.epoch1_checked:
                    print(f"COMPLETE gpu={gpu} recipe={job.recipe.name}; freeing gpu", flush=True)
                else:
                    print(f"FAILED gpu={gpu} recipe={job.recipe.name}; no epoch=001 result", flush=True)
                del active[gpu]

        mem = gpu_mem_mb()
        for gpu in GPU_IDS:
            if gpu in active or not queue:
                continue
            used = mem.get(gpu, 10**9)
            if used > FREE_MEMORY_MAX_MB:
                print(f"WAIT gpu={gpu} mem={used}MB busy", flush=True)
                continue
            recipe = queue.pop(0)
            job = launch(recipe, gpu)
            active[gpu] = job
            print(f"LAUNCH gpu={gpu} recipe={recipe.name} session={job.session} log={job.log_path}", flush=True)

        if queue or active:
            time.sleep(POLL_SECONDS)

    print("safe_ctrl queue complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
