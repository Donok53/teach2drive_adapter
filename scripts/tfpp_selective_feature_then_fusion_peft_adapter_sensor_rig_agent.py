#!/usr/bin/env python3
"""Selective feature-then-fusion PEFT agent for shifted-sensor-only adapters."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tfpp_feature_then_fusion_peft_adapter_sensor_rig_agent import (  # noqa: E402
    FeatureThenFusionPeftAdapterSensorRigAgent,
)


def get_entry_point() -> str:
    return "SelectiveFeatureThenFusionPeftAdapterSensorRigAgent"


def _parse_layers(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    text = str(value).strip().lower()
    if text in {"", "all", "*", "none"}:
        return None
    if text.startswith("early:"):
        return tuple(range(max(0, int(text.split(":", 1)[1]))))
    return tuple(sorted({int(item.strip()) for item in text.split(",") if item.strip()}))


class SelectiveFeatureThenFusionPeftAdapterSensorRigAgent(FeatureThenFusionPeftAdapterSensorRigAgent):
    """Apply only the trained adapter paths recorded in checkpoint metadata.

    This keeps camera-only, early-only, or fusion-disabled checkpoints from
    accidentally applying randomly initialized unused adapter branches at
    closed-loop evaluation time.
    """

    def _load_adapter(self) -> None:
        super()._load_adapter()

        checkpoint_path = (
            os.environ.get("TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT")
            or os.environ.get("TFPP_ADAPTER_CHECKPOINT")
            or ""
        )
        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=self.device)
        metadata = checkpoint.get("metadata", {})
        args = metadata.get("args", {})
        self._selective_stage_layers = _parse_layers(metadata.get("stage_adapter_layers", args.get("stage_adapter_layers", "all")))
        self._selective_modalities = str(metadata.get("stage_adapter_modalities", args.get("stage_adapter_modalities", "all")))
        self._selective_fusion_enabled = bool(metadata.get("fusion_adapter_enabled", not bool(args.get("disable_fusion_adapter", False))))
        print(
            "[SelectiveFeatureThenFusionPeftAdapterSensorRigAgent] selective "
            f"layers={self._selective_stage_layers if self._selective_stage_layers is not None else 'all'} "
            f"modalities={self._selective_modalities} fusion={int(self._selective_fusion_enabled)}",
            flush=True,
        )

    def _stage_enabled(self, layer_idx: int) -> bool:
        layers = getattr(self, "_selective_stage_layers", None)
        if layers is not None and int(layer_idx) not in layers:
            return False
        return str(getattr(self, "_selective_modalities", "all")) != "none"

    def _adapt_stage_pair(self, layer_idx: int, image_embd_layer: torch.Tensor, lidar_embd_layer: torch.Tensor):
        if not self._stage_enabled(int(layer_idx)):
            return image_embd_layer, lidar_embd_layer

        image_name = f"layer_{int(layer_idx)}_image"
        lidar_name = f"layer_{int(layer_idx)}_lidar"
        expected_image = self._stage_feature_shapes.get(image_name)
        expected_lidar = self._stage_feature_shapes.get(lidar_name)
        got_image = tuple(int(v) for v in image_embd_layer.shape[1:])
        got_lidar = tuple(int(v) for v in lidar_embd_layer.shape[1:])
        if got_image != expected_image or got_lidar != expected_lidar:
            key = f"{image_name}:{got_image}:{got_lidar}"
            warned = getattr(self, "_stage_shape_warnings", set())
            if key not in warned:
                warned.add(key)
                self._stage_shape_warnings = warned
                print(
                    "[SelectiveFeatureThenFusionPeftAdapterSensorRigAgent] stage shape mismatch "
                    f"layer={layer_idx} image={got_image}/{expected_image} lidar={got_lidar}/{expected_lidar}; "
                    "skipping stage adapter",
                    flush=True,
                )
            return image_embd_layer, lidar_embd_layer

        adapted_image, adapted_lidar = self._adapter.adapt_layer(
            int(layer_idx),
            image_embd_layer.float(),
            lidar_embd_layer.float(),
        )
        modality = str(getattr(self, "_selective_modalities", "all"))
        if modality not in {"all", "camera"}:
            adapted_image = image_embd_layer.float()
        if modality not in {"all", "lidar"}:
            adapted_lidar = lidar_embd_layer.float()
        adapted_image = adapted_image.to(dtype=image_embd_layer.dtype)
        adapted_lidar = adapted_lidar.to(dtype=lidar_embd_layer.dtype)
        blend = float(self._stage_blend)
        if blend < 1.0:
            adapted_image = image_embd_layer + blend * (adapted_image - image_embd_layer)
            adapted_lidar = lidar_embd_layer + blend * (adapted_lidar - lidar_embd_layer)
        return adapted_image, adapted_lidar

    def _adapt_fused(self, fused: torch.Tensor, net_index: int) -> torch.Tensor:
        if not bool(getattr(self, "_selective_fusion_enabled", True)):
            return fused
        return super()._adapt_fused(fused, net_index)


if __name__ == "__main__":
    print("This file is a CARLA leaderboard agent module.", file=sys.stderr)
