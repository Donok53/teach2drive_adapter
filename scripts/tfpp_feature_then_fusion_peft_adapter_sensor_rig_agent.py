#!/usr/bin/env python3
"""Feature-then-fusion adapter agent with optional target-only LoRA PEFT state."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tfpp_feature_then_fusion_adapter_sensor_rig_agent import (  # noqa: E402
    FeatureThenFusionAdapterSensorRigAgent,
    _ensure_adapter_import_path,
)


def get_entry_point() -> str:
    return "FeatureThenFusionPeftAdapterSensorRigAgent"


class FeatureThenFusionPeftAdapterSensorRigAgent(FeatureThenFusionAdapterSensorRigAgent):
    def _load_adapter(self) -> None:
        super()._load_adapter()

        checkpoint_path = (
            os.environ.get("TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT")
            or os.environ.get("TFPP_ADAPTER_CHECKPOINT")
            or ""
        )
        if not checkpoint_path:
            return

        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=self.device)
        metadata = checkpoint.get("metadata", {})
        peft_lora = metadata.get("peft_lora", {})
        peft_state = checkpoint.get("peft_lora_state", {})
        rank = int(peft_lora.get("rank", 0) or 0)
        if rank <= 0 or not peft_state:
            print("[FeatureThenFusionPeftAdapterSensorRigAgent] peft_lora=off", flush=True)
            return

        _ensure_adapter_import_path()
        from teach2drive_adapter.peft_lora import install_lora_adapters, load_lora_state_dict

        modules_per_net: list[list[str]] = []
        load_infos: list[dict[str, list[str]]] = []
        for net in self.nets:
            modules = install_lora_adapters(
                net,
                include=str(peft_lora.get("include", "")),
                exclude=str(peft_lora.get("exclude", "")),
                rank=rank,
                alpha=float(peft_lora.get("alpha", 16.0)),
                dropout=0.0,
            )
            load_info = load_lora_state_dict(net, peft_state, strict=False)
            modules_per_net.append(modules)
            load_infos.append(load_info)

        first_modules = modules_per_net[0] if modules_per_net else []
        unexpected = sum(len(info.get("unexpected", [])) for info in load_infos)
        missing = sum(len(info.get("missing", [])) for info in load_infos)
        print(
            "[FeatureThenFusionPeftAdapterSensorRigAgent] peft_lora=on "
            f"rank={rank} alpha={float(peft_lora.get('alpha', 16.0)):.3f} "
            f"nets={len(modules_per_net)} modules={len(first_modules)} "
            f"missing={missing} unexpected={unexpected}",
            flush=True,
        )
