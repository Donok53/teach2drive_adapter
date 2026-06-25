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
        self._load_output_residual(checkpoint, metadata)
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

    def _load_output_residual(self, checkpoint, metadata) -> None:
        residual_meta = metadata.get("output_residual", {})
        aux_state = checkpoint.get("aux_state", {})
        residual_state = aux_state.get("output_residual_head", {})
        if not bool(residual_meta.get("enabled", False)) or not residual_state:
            self._output_residual_head = None
            print("[FeatureThenFusionPeftAdapterSensorRigAgent] output_residual=off", flush=True)
            return

        _ensure_adapter_import_path()
        from teach2drive_adapter.train_transfuserpp_task_feature_adapter import GatedOutputResidualHead

        self._output_residual_head = GatedOutputResidualHead(
            fused_channels=int(self._fused_feature_shape[0]),
            checkpoint_dim=int(residual_meta.get("checkpoint_dim", 20)),
            speed_classes=int(residual_meta.get("speed_classes", len(getattr(self.config, "target_speeds", [])) or 1)),
            hidden_dim=int(residual_meta.get("hidden_dim", 256)),
            checkpoint_scale=float(residual_meta.get("checkpoint_scale", 0.75)),
            speed_logit_scale=float(residual_meta.get("speed_logit_scale", 1.5)),
            gate_bias=float(residual_meta.get("gate_bias", -2.0)),
            gate_max=float(residual_meta.get("gate_max", 1.0)),
            dropout=float(residual_meta.get("dropout", 0.0)),
            controller_residual=bool(residual_meta.get("controller_residual", False)),
            controller_lateral_scale=float(residual_meta.get("controller_lateral_scale", 1.0)),
            controller_progress_scale=float(residual_meta.get("controller_progress_scale", 1.0)),
            controller_speed_logit_scale=float(residual_meta.get("controller_speed_logit_scale", 2.0)),
            controller_gate_bias=float(residual_meta.get("controller_gate_bias", -2.0)),
            controller_gate_max=float(residual_meta.get("controller_gate_max", 1.0)),
            speed_class_values=residual_meta.get("speed_class_values"),
        ).to(self.device)
        missing, unexpected = self._output_residual_head.load_state_dict(residual_state, strict=False)
        self._output_residual_head.eval()
        self._patch_output_residual_for_nets()
        print(
            "[FeatureThenFusionPeftAdapterSensorRigAgent] output_residual=on "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"checkpoint_scale={float(residual_meta.get('checkpoint_scale', 0.75)):.3f} "
            f"speed_logit_scale={float(residual_meta.get('speed_logit_scale', 1.5)):.3f} "
            f"gate_max={float(residual_meta.get('gate_max', 1.0)):.3f} "
            f"controller={int(bool(residual_meta.get('controller_residual', False)))}",
            flush=True,
        )

    def _patch_output_residual_for_nets(self) -> None:
        if getattr(self, "_output_residual_head", None) is None:
            return
        for index, net in enumerate(self.nets):
            original_forward = net.forward

            def residual_forward(*args, _original_forward=original_forward, _index=index, **kwargs):
                output = _original_forward(*args, **kwargs)
                if not isinstance(output, (tuple, list)) or len(output) < 3:
                    return output
                fused = getattr(self, "_last_fused_by_net", {}).get(int(_index))
                if fused is None:
                    return output
                pred_target_speed = output[1]
                pred_checkpoint = output[2]
                with torch.no_grad():
                    adapted_checkpoint, adapted_speed, _ = self._output_residual_head(
                        fused,
                        pred_checkpoint,
                        pred_target_speed,
                    )
                out = list(output)
                if adapted_speed is not None:
                    out[1] = adapted_speed
                if adapted_checkpoint is not None:
                    out[2] = adapted_checkpoint
                return tuple(out) if isinstance(output, tuple) else out

            net.forward = residual_forward
