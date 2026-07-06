#!/usr/bin/env python3
"""Idempotent in-place patch: add item-1 ego-speed input augmentation to
teach2drive_adapter/transfuserpp_bridge.py.

Deliberately anchor-based (imports block + the `"ego_vel": scalar[:, :1]...` line)
so it applies to whatever version of the file is on disk -- the gist working tree
is diverged from remote main, so we edit its actual file rather than overwrite it.

Usage: python scripts/patch_ego_speed_aug.py [path_to_transfuserpp_bridge.py]
"""
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "teach2drive_adapter" / "transfuserpp_bridge.py"
path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
src = path.read_text(encoding="utf-8")

if "_augment_ego_vel" in src:
    print(f"already patched: {path}")
    sys.exit(0)

IMPORT_ANCHOR = "from torch.nn import functional as F\n"
HELPER = '''
import os as _os

# Item-1 (anti causal-confusion / "inertia"): ego-speed input augmentation,
# applied during TRAINING ONLY (gated by torch.is_grad_enabled -> the prior
# forward, validation, and the closed-loop agent are all no_grad and untouched).
# Env-controlled; defaults 0.0 -> fully backward compatible.
_EGO_SPEED_NOISE_STD = float(_os.environ.get("T2D_EGO_SPEED_NOISE_STD", "0") or 0.0)
_EGO_SPEED_DROPOUT_P = float(_os.environ.get("T2D_EGO_SPEED_DROPOUT_P", "0") or 0.0)


def _augment_ego_vel(ego_vel):
    """Perturb ego-speed so the policy can't use its own speed as a
    'if slow -> stay stopped' shortcut. Gaussian noise + random dropout-to-zero."""
    if not torch.is_grad_enabled():
        return ego_vel
    if _EGO_SPEED_NOISE_STD <= 0.0 and _EGO_SPEED_DROPOUT_P <= 0.0:
        return ego_vel
    out = ego_vel
    if _EGO_SPEED_NOISE_STD > 0.0:
        out = out + torch.randn_like(out) * _EGO_SPEED_NOISE_STD
    if _EGO_SPEED_DROPOUT_P > 0.0:
        keep = (torch.rand_like(out) >= _EGO_SPEED_DROPOUT_P).to(out.dtype)
        out = out * keep
    return out.clamp_min(0.0)

'''

EGO_OLD = '"ego_vel": scalar[:, :1].contiguous(),'
EGO_NEW = '"ego_vel": _augment_ego_vel(scalar[:, :1].contiguous()),'

if IMPORT_ANCHOR not in src:
    print("FATAL: import anchor not found", file=sys.stderr)
    sys.exit(2)
if EGO_OLD not in src:
    print("FATAL: ego_vel anchor not found", file=sys.stderr)
    sys.exit(2)

src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + HELPER, 1)
src = src.replace(EGO_OLD, EGO_NEW, 1)

# sanity: still valid python
import ast
ast.parse(src)

path.write_text(src, encoding="utf-8")
print(f"patched OK: {path}")
