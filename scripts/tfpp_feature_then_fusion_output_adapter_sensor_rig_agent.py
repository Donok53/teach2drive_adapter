#!/usr/bin/env python3
"""TransFuser++ agent with feature+fusion adapters followed by an output adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tfpp_adapter_sensor_rig_agent import AdapterSensorRigAgent, _env_float, _ensure_adapter_import_path


def get_entry_point() -> str:
    return "FeatureThenFusionOutputAdapterSensorRigAgent"


class FeatureThenFusionOutputAdapterSensorRigAgent(AdapterSensorRigAgent):
    """Apply the learned feature+fusion correction before the learned output adapter."""

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index=route_index, traffic_manager=traffic_manager)
        self._load_feature_then_fusion_adapter()
        self._patch_backbones_with_feature_then_fusion_adapter()

    def _load_feature_then_fusion_adapter(self) -> None:
        checkpoint_path = os.environ.get("TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT", "")
        if not checkpoint_path:
            raise ValueError("TFPP_FEATURE_THEN_FUSION_ADAPTER_CHECKPOINT is required")

        _ensure_adapter_import_path()
        from teach2drive_adapter.train_transfuserpp_feature_then_fusion_adapter import (
            ExtrinsicAwareFeatureThenFusionAdapter,
            FeatureThenFusionAdapter,
            build_extrinsic_vector,
        )

        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=self.device)
        metadata = checkpoint.get("metadata", {})
        args = metadata.get("args", {})
        stage_shapes_raw = checkpoint.get("stage_feature_shapes", metadata.get("stage_feature_shapes", {}))
        fused_shape_raw = checkpoint.get("fused_feature_shape", metadata.get("fused_feature_shape", []))
        self._ff_stage_feature_shapes = {
            str(name): tuple(int(v) for v in shape) for name, shape in stage_shapes_raw.items()
        }
        self._ff_fused_feature_shape = tuple(int(v) for v in fused_shape_raw)
        if not self._ff_stage_feature_shapes or len(self._ff_fused_feature_shape) != 3:
            raise ValueError("Invalid feature-then-fusion adapter shapes in checkpoint")

        self._ff_extrinsic_aware = (
            bool(metadata.get("extrinsic_aware", False))
            or metadata.get("mode") == "transfuserpp_extrinsic_feature_then_fusion_adapter"
        )
        if self._ff_extrinsic_aware:
            extrinsic_vector = metadata.get("extrinsic_vector") or build_extrinsic_vector(
                str(metadata.get("source_profile", "front_triplet_shifted"))
            )
            self._ff_adapter = ExtrinsicAwareFeatureThenFusionAdapter(
                stage_feature_shapes=self._ff_stage_feature_shapes,
                fused_feature_shape=self._ff_fused_feature_shape,
                extrinsic_vector=extrinsic_vector,
                hidden_channels=int(args.get("hidden_channels", 0)),
                blocks=int(args.get("blocks", 2)),
                dropout=float(args.get("dropout", 0.0)),
                extrinsic_hidden_dim=int(args.get("extrinsic_hidden_dim", 64)),
                extrinsic_dropout=float(args.get("extrinsic_dropout", 0.0)),
            ).to(self.device)
        else:
            self._ff_adapter = FeatureThenFusionAdapter(
                stage_feature_shapes=self._ff_stage_feature_shapes,
                fused_feature_shape=self._ff_fused_feature_shape,
                hidden_channels=int(args.get("hidden_channels", 0)),
                blocks=int(args.get("blocks", 2)),
                dropout=float(args.get("dropout", 0.0)),
            ).to(self.device)

        missing, unexpected = self._ff_adapter.load_state_dict(checkpoint["model_state"], strict=False)
        self._ff_adapter.eval()
        shared_blend = _env_float("TFPP_FEATURE_ADAPTER_BLEND", 1.0)
        self._ff_stage_blend = _env_float("TFPP_STAGE_FEATURE_ADAPTER_BLEND", shared_blend)
        self._ff_fusion_blend = _env_float("TFPP_FUSION_ADAPTER_BLEND", shared_blend)
        print(
            "[FeatureThenFusionOutputAdapterSensorRigAgent] loaded feature+fusion adapter "
            f"checkpoint={Path(checkpoint_path).expanduser()} "
            f"extrinsic_aware={self._ff_extrinsic_aware} "
            f"stage_blend={self._ff_stage_blend:.3f} fusion_blend={self._ff_fusion_blend:.3f} "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    def _adapt_stage_pair(self, layer_idx: int, image_embd_layer: torch.Tensor, lidar_embd_layer: torch.Tensor):
        image_name = f"layer_{int(layer_idx)}_image"
        lidar_name = f"layer_{int(layer_idx)}_lidar"
        expected_image = self._ff_stage_feature_shapes.get(image_name)
        expected_lidar = self._ff_stage_feature_shapes.get(lidar_name)
        got_image = tuple(int(v) for v in image_embd_layer.shape[1:])
        got_lidar = tuple(int(v) for v in lidar_embd_layer.shape[1:])
        if got_image != expected_image or got_lidar != expected_lidar:
            key = f"{image_name}:{got_image}:{got_lidar}"
            warned = getattr(self, "_ff_stage_shape_warnings", set())
            if key not in warned:
                warned.add(key)
                self._ff_stage_shape_warnings = warned
                print(
                    "[FeatureThenFusionOutputAdapterSensorRigAgent] stage shape mismatch "
                    f"layer={layer_idx} image={got_image}/{expected_image} "
                    f"lidar={got_lidar}/{expected_lidar}; skipping stage adapter",
                    flush=True,
                )
            return image_embd_layer, lidar_embd_layer

        adapted_image, adapted_lidar = self._ff_adapter.adapt_layer(
            int(layer_idx),
            image_embd_layer.float(),
            lidar_embd_layer.float(),
        )
        adapted_image = adapted_image.to(dtype=image_embd_layer.dtype)
        adapted_lidar = adapted_lidar.to(dtype=lidar_embd_layer.dtype)
        blend = float(self._ff_stage_blend)
        if blend < 1.0:
            adapted_image = image_embd_layer + blend * (adapted_image - image_embd_layer)
            adapted_lidar = lidar_embd_layer + blend * (adapted_lidar - lidar_embd_layer)
        return adapted_image, adapted_lidar

    def _adapt_fused(self, fused: torch.Tensor, net_index: int) -> torch.Tensor:
        if fused.ndim != 4:
            return fused
        got = tuple(int(v) for v in fused.shape[1:])
        if got != self._ff_fused_feature_shape:
            if getattr(self, "_ff_fused_shape_warned", False) is False:
                print(
                    "[FeatureThenFusionOutputAdapterSensorRigAgent] fused feature shape mismatch "
                    f"net={net_index} got={got} expected={self._ff_fused_feature_shape}; skipping fusion adapter",
                    flush=True,
                )
                self._ff_fused_shape_warned = True
            return fused

        adapted = self._ff_adapter.adapt_fused(fused.float()).to(dtype=fused.dtype)
        blend = float(self._ff_fusion_blend)
        if blend < 1.0:
            adapted = fused + blend * (adapted - fused)
        return adapted

    def _patch_backbones_with_feature_then_fusion_adapter(self) -> None:
        for index, net in enumerate(self.nets):
            backbone = net.backbone
            original_forward = backbone.forward

            def adapted_fuse_features(image_features, lidar_features, layer_idx, _backbone=backbone):
                idx = int(layer_idx)
                image_embd_layer = _backbone.avgpool_img(image_features)
                lidar_embd_layer = _backbone.avgpool_lidar(lidar_features)
                lidar_embd_layer = _backbone.lidar_channel_to_img[idx](lidar_embd_layer)

                image_embd_layer, lidar_embd_layer = self._adapt_stage_pair(idx, image_embd_layer, lidar_embd_layer)
                image_features_layer, lidar_features_layer = _backbone.transformers[idx](image_embd_layer, lidar_embd_layer)
                lidar_features_layer = _backbone.img_channel_to_lidar[idx](lidar_features_layer)

                image_features_layer = F.interpolate(
                    image_features_layer,
                    size=(image_features.shape[2], image_features.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                if _backbone.lidar_video:
                    lidar_features_layer = F.interpolate(
                        lidar_features_layer,
                        size=(lidar_features.shape[2], lidar_features.shape[3], lidar_features.shape[4]),
                        mode="trilinear",
                        align_corners=False,
                    )
                else:
                    lidar_features_layer = F.interpolate(
                        lidar_features_layer,
                        size=(lidar_features.shape[2], lidar_features.shape[3]),
                        mode="bilinear",
                        align_corners=False,
                    )
                return image_features + image_features_layer, lidar_features + lidar_features_layer

            backbone.fuse_features = adapted_fuse_features

            def adapted_forward(*args, _original_forward=original_forward, _index=index, **kwargs):
                output = _original_forward(*args, **kwargs)
                if not isinstance(output, (tuple, list)) or len(output) < 2:
                    return output
                adapted = self._adapt_fused(output[1], _index)
                if isinstance(output, tuple):
                    return (output[0], adapted, *output[2:])
                out = list(output)
                out[1] = adapted
                return out

            backbone.forward = adapted_forward

        print(
            f"[FeatureThenFusionOutputAdapterSensorRigAgent] patched {len(self.nets)} backbone(s)",
            flush=True,
        )


if __name__ == "__main__":
    print("This file is a CARLA leaderboard agent module.", file=sys.stderr)
