import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import Teach2DriveIndexDataset
from .transfuserpp_bridge import base_target_from_checkpoint, load_transfuserpp, prepare_transfuserpp_inputs, speed_expectation


def _camera_list(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def _flatten_checkpoint(pred_checkpoint: torch.Tensor, width: int) -> torch.Tensor:
    flat = torch.zeros((pred_checkpoint.shape[0], width), dtype=pred_checkpoint.dtype, device=pred_checkpoint.device)
    raw = pred_checkpoint.reshape(pred_checkpoint.shape[0], -1)
    flat[:, : min(width, raw.shape[1])] = raw[:, : flat.shape[1]]
    return flat


def _load_feature_then_fusion_adapter(checkpoint_path: str, device: torch.device):
    if not checkpoint_path:
        return None, None
    from .train_transfuserpp_feature_then_fusion_adapter import (
        ExtrinsicAwareFeatureThenFusionAdapter,
        FeatureThenFusionAdapter,
        build_extrinsic_vector,
    )

    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device)
    metadata = checkpoint.get("metadata", {})
    ckpt_args = metadata.get("args", {})
    stage_shapes = {
        str(name): tuple(int(v) for v in shape)
        for name, shape in checkpoint.get("stage_feature_shapes", metadata.get("stage_feature_shapes", {})).items()
    }
    fused_shape = tuple(int(v) for v in checkpoint.get("fused_feature_shape", metadata.get("fused_feature_shape", [])))
    if not stage_shapes or len(fused_shape) != 3:
        raise ValueError(f"Invalid feature-then-fusion adapter checkpoint shapes: {checkpoint_path}")
    extrinsic_aware = bool(metadata.get("extrinsic_aware", False)) or metadata.get("mode") == "transfuserpp_extrinsic_feature_then_fusion_adapter"
    if extrinsic_aware:
        extrinsic_vector = metadata.get("extrinsic_vector") or build_extrinsic_vector(str(metadata.get("source_profile", "front_triplet_shifted")))
        adapter = ExtrinsicAwareFeatureThenFusionAdapter(
            stage_feature_shapes=stage_shapes,
            fused_feature_shape=fused_shape,
            extrinsic_vector=extrinsic_vector,
            hidden_channels=int(ckpt_args.get("hidden_channels", 0)),
            blocks=int(ckpt_args.get("blocks", 2)),
            dropout=float(ckpt_args.get("dropout", 0.0)),
            extrinsic_hidden_dim=int(ckpt_args.get("extrinsic_hidden_dim", 64)),
            extrinsic_dropout=float(ckpt_args.get("extrinsic_dropout", 0.0)),
        ).to(device)
    else:
        adapter = FeatureThenFusionAdapter(
            stage_feature_shapes=stage_shapes,
            fused_feature_shape=fused_shape,
            hidden_channels=int(ckpt_args.get("hidden_channels", 0)),
            blocks=int(ckpt_args.get("blocks", 2)),
            dropout=float(ckpt_args.get("dropout", 0.0)),
        ).to(device)
    adapter._stage_feature_shapes = stage_shapes  # type: ignore[attr-defined]  # pylint: disable=protected-access
    adapter._fused_feature_shape = fused_shape  # type: ignore[attr-defined]  # pylint: disable=protected-access
    missing, unexpected = adapter.load_state_dict(checkpoint["model_state"], strict=False)
    adapter.eval()
    info = {
        "checkpoint": str(Path(checkpoint_path).expanduser()),
        "stage_feature_shapes": {key: list(value) for key, value in stage_shapes.items()},
        "fused_feature_shape": list(fused_shape),
        "extrinsic_aware": extrinsic_aware,
        "missing": len(missing),
        "unexpected": len(unexpected),
    }
    return adapter, info


def _patch_feature_then_fusion_backbone(net, adapter, stage_blend: float, fusion_blend: float):
    if adapter is None:
        return
    backbone = net.backbone
    backbone.t2d_feature_then_fusion_adapter = adapter
    backbone.t2d_stage_feature_shapes = {
        key: tuple(int(v) for v in value)
        for key, value in adapter._stage_feature_shapes.items()  # pylint: disable=protected-access
    }
    backbone.t2d_fused_feature_shape = tuple(int(v) for v in adapter._fused_feature_shape)  # pylint: disable=protected-access
    backbone.t2d_stage_feature_adapter_blend = float(stage_blend)
    backbone.t2d_fusion_adapter_blend = float(fusion_blend)

    backbone_class = backbone.__class__
    if getattr(backbone_class, "_t2d_feature_then_fusion_patched", False):
        return
    backbone_class._t2d_original_forward = backbone_class.forward

    def adapt_stage_pair(self, layer_idx: int, image_embd_layer: torch.Tensor, lidar_embd_layer: torch.Tensor):
        adapter_module = self.t2d_feature_then_fusion_adapter
        image_name = f"layer_{int(layer_idx)}_image"
        lidar_name = f"layer_{int(layer_idx)}_lidar"
        expected_image = self.t2d_stage_feature_shapes[image_name]
        expected_lidar = self.t2d_stage_feature_shapes[lidar_name]
        got_image = tuple(int(v) for v in image_embd_layer.shape[1:])
        got_lidar = tuple(int(v) for v in lidar_embd_layer.shape[1:])
        if got_image != expected_image or got_lidar != expected_lidar:
            raise ValueError(
                f"feature-then-fusion stage shape mismatch layer={layer_idx}: "
                f"image {got_image} != {expected_image}, lidar {got_lidar} != {expected_lidar}"
            )
        adapted_image, adapted_lidar = adapter_module.adapt_layer(
            int(layer_idx),
            image_embd_layer.float(),
            lidar_embd_layer.float(),
        )
        adapted_image = adapted_image.to(dtype=image_embd_layer.dtype)
        adapted_lidar = adapted_lidar.to(dtype=lidar_embd_layer.dtype)
        blend = float(self.t2d_stage_feature_adapter_blend)
        if blend < 1.0:
            adapted_image = image_embd_layer + blend * (adapted_image - image_embd_layer)
            adapted_lidar = lidar_embd_layer + blend * (adapted_lidar - lidar_embd_layer)
        return adapted_image, adapted_lidar

    def adapted_fuse_features(self, image_features, lidar_features, layer_idx):
        idx = int(layer_idx)
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)
        lidar_embd_layer = self.lidar_channel_to_img[idx](lidar_embd_layer)
        image_embd_layer, lidar_embd_layer = adapt_stage_pair(self, idx, image_embd_layer, lidar_embd_layer)
        image_features_layer, lidar_features_layer = self.transformers[idx](image_embd_layer, lidar_embd_layer)
        lidar_features_layer = self.img_channel_to_lidar[idx](lidar_features_layer)
        image_features_layer = torch.nn.functional.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        if self.lidar_video:
            lidar_features_layer = torch.nn.functional.interpolate(
                lidar_features_layer,
                size=(lidar_features.shape[2], lidar_features.shape[3], lidar_features.shape[4]),
                mode="trilinear",
                align_corners=False,
            )
        else:
            lidar_features_layer = torch.nn.functional.interpolate(
                lidar_features_layer,
                size=(lidar_features.shape[2], lidar_features.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
        return image_features + image_features_layer, lidar_features + lidar_features_layer

    def adapted_forward(self, *args, **kwargs):
        output = self.__class__._t2d_original_forward(self, *args, **kwargs)
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return output
        fused = output[1]
        expected = self.t2d_fused_feature_shape
        got = tuple(int(v) for v in fused.shape[1:])
        if got != expected:
            raise ValueError(f"feature-then-fusion fused shape mismatch: {got} != {expected}")
        adapted = self.t2d_feature_then_fusion_adapter.adapt_fused(fused.float()).to(dtype=fused.dtype)
        blend = float(self.t2d_fusion_adapter_blend)
        if blend < 1.0:
            adapted = fused + blend * (adapted - fused)
        if isinstance(output, tuple):
            return (output[0], adapted, *output[2:])
        out = list(output)
        out[1] = adapted
        return out

    backbone_class.fuse_features = adapted_fuse_features
    backbone_class.forward = adapted_forward
    backbone_class._t2d_feature_then_fusion_patched = True


def build_cache(args: argparse.Namespace) -> None:
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    cameras = _camera_list(args.cameras)
    indices = None
    if args.max_samples > 0:
        indices = np.arange(args.max_samples, dtype=np.int64)
    dataset = Teach2DriveIndexDataset(
        args.index,
        indices=indices,
        cameras=cameras,
        image_size=tuple(args.image_size),
        lidar_size=args.lidar_size,
        episode_root_override=args.episode_root_override,
        teacher_view_root=(args.teacher_view_root or None),
        teacher_view_dirname=args.teacher_view_dirname,
        teacher_view_camera=args.tfpp_camera,
    )
    _use_teacher_view = bool(args.teacher_view_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    net, config, load_info = load_transfuserpp(args.garage_root, args.team_config, device=device, checkpoint=args.checkpoint)
    feature_then_fusion_adapter, feature_then_fusion_info = _load_feature_then_fusion_adapter(
        args.feature_then_fusion_adapter_checkpoint,
        device,
    )
    if feature_then_fusion_adapter is not None:
        _patch_feature_then_fusion_backbone(
            net,
            feature_then_fusion_adapter,
            stage_blend=float(args.stage_feature_adapter_blend),
            fusion_blend=float(args.fusion_adapter_blend),
        )
        print(
            json.dumps(
                {
                    "feature_then_fusion_prior": True,
                    "stage_feature_adapter_blend": float(args.stage_feature_adapter_blend),
                    "fusion_adapter_blend": float(args.fusion_adapter_blend),
                    "adapter": feature_then_fusion_info,
                },
                indent=2,
            ),
            flush=True,
        )
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs for prior caching", flush=True)
        net = nn.DataParallel(net)
    net.eval()

    chunks = {
        "sample_index": [],
        "sample_episode": [],
        "sample_frame": [],
        "scalar": [],
        "layout": [],
        "target": [],
        "stop_state": [],
        "stop_reason": [],
        "stop_reason_mask": [],
        "sample_weight": [],
        "base_target": [],
        "checkpoint_flat": [],
        "speed_logits": [],
        "expected_speed": [],
    }
    checkpoint_width = int(getattr(config, "predict_checkpoint_len", 10)) * 2
    speed_classes = len(getattr(config, "target_speeds", []))
    start = time.time()
    seen = 0
    _tfpp_idx = cameras.index(args.tfpp_camera) if args.tfpp_camera in cameras else 0
    for step, batch in enumerate(loader, start=1):
        scalar = batch["scalar"].to(device, non_blocking=True)
        camera = batch["camera"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        if _use_teacher_view and "camera_teacher" in batch:
            # replace the tfpp_camera channel with the reprojected x=-1.5 teacher view
            camera_teacher = batch["camera_teacher"].to(device, non_blocking=True)  # [B,1,3,H,W]
            camera = camera.clone()
            camera[:, _tfpp_idx] = camera_teacher[:, 0]
        inputs = prepare_transfuserpp_inputs(
            scalar=scalar,
            camera=camera,
            lidar=lidar,
            cameras=cameras,
            config=config,
            command_mode=args.command_mode,
            tfpp_camera=args.tfpp_camera,
        )
        with torch.no_grad():
            outputs = net(**inputs)
            pred_target_speed = outputs[1]
            pred_checkpoint = outputs[2]
            base_target = base_target_from_checkpoint(
                pred_checkpoint=pred_checkpoint,
                pred_target_speed=pred_target_speed,
                scalar=scalar,
                config=config,
                target_dim=batch["target"].shape[1],
                speed_dim=args.speed_dim,
            )
            if pred_checkpoint is None:
                checkpoint_flat = torch.zeros((scalar.shape[0], checkpoint_width), dtype=scalar.dtype, device=device)
            else:
                checkpoint_flat = _flatten_checkpoint(pred_checkpoint, checkpoint_width)
            if pred_target_speed is None:
                speed_logits = torch.zeros((scalar.shape[0], speed_classes), dtype=scalar.dtype, device=device)
            else:
                speed_logits = pred_target_speed
            expected_speed = speed_expectation(pred_target_speed, config, scalar.shape[0], device)

        chunks["sample_index"].append(batch["index"].numpy().astype(np.int64))
        chunks["sample_episode"].append(batch["episode_idx"].numpy().astype(np.int64))
        chunks["sample_frame"].append(batch["frame_idx"].numpy().astype(np.int64))
        chunks["scalar"].append(batch["scalar"].numpy().astype(np.float32))
        chunks["layout"].append(batch["layout"].numpy().astype(np.float32))
        chunks["target"].append(batch["target"].numpy().astype(np.float32))
        chunks["stop_state"].append(batch["stop_state"].numpy().astype(np.int64))
        chunks["stop_reason"].append(batch["stop_reason"].numpy().astype(np.int64))
        chunks["stop_reason_mask"].append(batch["stop_reason_mask"].numpy().astype(np.float32))
        chunks["sample_weight"].append(batch["sample_weight"].numpy().astype(np.float32))
        chunks["base_target"].append(base_target.detach().cpu().numpy().astype(np.float32))
        chunks["checkpoint_flat"].append(checkpoint_flat.detach().cpu().numpy().astype(np.float32))
        chunks["speed_logits"].append(speed_logits.detach().cpu().numpy().astype(np.float32))
        chunks["expected_speed"].append(expected_speed.detach().cpu().numpy().astype(np.float32))

        seen += int(scalar.shape[0])
        if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
            elapsed = max(time.time() - start, 1e-6)
            print(f"cache_step={step:05d}/{len(loader):05d} samples={seen} samples/s={seen/elapsed:.1f}", flush=True)

    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    metadata = {
        "index": str(Path(args.index).expanduser()),
        "episode_root_override": args.episode_root_override,
        "garage_root": str(Path(args.garage_root).expanduser()),
        "team_config": str(Path(args.team_config).expanduser()),
        "cameras": cameras,
        "tfpp_camera": args.tfpp_camera,
        "command_mode": args.command_mode,
        "samples": int(len(arrays["sample_index"])),
        "load_info": load_info,
        "feature_then_fusion_adapter": feature_then_fusion_info,
        "stage_feature_adapter_blend": float(args.stage_feature_adapter_blend),
        "fusion_adapter_blend": float(args.fusion_adapter_blend),
        "target_speeds": list(getattr(config, "target_speeds", [])),
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata), dtype=object)
    np.savez(out_path, **arrays)
    (out_path.with_suffix(".json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), **metadata}, indent=2), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache frozen TransFuser++ priors for fast Teach2Drive adapter training.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--garage-root", required=True)
    parser.add_argument("--team-config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--feature-then-fusion-adapter-checkpoint", default="")
    parser.add_argument("--stage-feature-adapter-blend", type=float, default=1.0)
    parser.add_argument("--fusion-adapter-blend", type=float, default=1.0)
    parser.add_argument("--episode-root-override", default="")
    parser.add_argument("--cameras", default="front,left,right")
    parser.add_argument("--tfpp-camera", default="front")
    # v5: build the TEACHER prior cache from the reprojected x=-1.5 view instead of
    # the real (shifted) camera. When set, the tfpp_camera channel is replaced by
    # the reprojected view loaded from <teacher-view-root>/<source_route>/<dirname>/<step>.jpg
    parser.add_argument("--teacher-view-root", default="")
    parser.add_argument("--teacher-view-dirname", default="rgb_front_teacher_xm15")
    parser.add_argument("--command-mode", choices=["lane_follow", "target_angle"], default="target_angle")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--lidar-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--speed-dim", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    build_cache(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
