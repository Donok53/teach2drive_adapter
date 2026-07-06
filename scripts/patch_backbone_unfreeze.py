#!/usr/bin/env python3
"""Idempotent in-place patch: add env-controlled backbone unfreeze to
teach2drive_adapter/train_transfuserpp_task_feature_adapter.py.

The gist trainer builds its optimizer as
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad), ...)
so simply flipping requires_grad on the perception backbone params BEFORE that
line makes them join the optimizer automatically -- no optimizer surgery.

Enables v3: adapt the frozen TF++ perception backbone (self.backbone, the image
+ lidar CNN encoders) to the new camera geometry, on single-rig target data.
Controlled by T2D_UNFREEZE_INCLUDE / T2D_UNFREEZE_EXCLUDE (comma regex list);
empty -> no-op (backward compatible).

Usage: python scripts/patch_backbone_unfreeze.py [path_to_trainer.py]
"""
import sys, ast
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "teach2drive_adapter" / "train_transfuserpp_task_feature_adapter.py"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
src = path.read_text(encoding="utf-8")

if "v3-unfreeze" in src:
    print(f"already patched: {path}")
    sys.exit(0)

ANCHOR = "    optimizer = torch.optim.AdamW("
if src.count(ANCHOR) != 1:
    print(f"FATAL: optimizer anchor found {src.count(ANCHOR)} times (need 1)", file=sys.stderr)
    sys.exit(2)

BLOCK = (
    "    # v3-unfreeze: adapt perception backbone to the new sensor geometry.\n"
    "    import os as _os_uf, re as _re_uf\n"
    "    _uf_inc = [p for p in _os_uf.environ.get('T2D_UNFREEZE_INCLUDE', '').split(',') if p.strip()]\n"
    "    _uf_exc = [p for p in _os_uf.environ.get('T2D_UNFREEZE_EXCLUDE', '').split(',') if p.strip()]\n"
    "    if _uf_inc:\n"
    "        _uf_n = 0\n"
    "        for _uf_name, _uf_p in model.named_parameters():\n"
    "            if any(_re_uf.search(p, _uf_name) for p in _uf_inc) and not any(_re_uf.search(e, _uf_name) for e in _uf_exc):\n"
    "                _uf_p.requires_grad_(True)\n"
    "                _uf_n += 1\n"
    "        print('[v3-unfreeze] requires_grad=True on %d params include=%r exclude=%r' % (_uf_n, _uf_inc, _uf_exc), flush=True)\n"
)

src = src.replace(ANCHOR, BLOCK + ANCHOR, 1)
ast.parse(src)  # sanity: still valid python
path.write_text(src, encoding="utf-8")
print(f"patched OK: {path}")
