from __future__ import annotations

import re
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen Linear layer with a small trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _split_patterns(raw: str | Iterable[str]) -> list[str]:
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [str(item).strip() for item in raw if str(item).strip()]


def _matches(name: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    return any(re.search(pattern, name) for pattern in patterns)


def _resolve_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def install_lora_adapters(
    root: nn.Module,
    include: str | Iterable[str],
    exclude: str | Iterable[str] = "",
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    """Replace matching Linear modules with LoRA-wrapped Linear modules."""

    include_patterns = _split_patterns(include)
    exclude_patterns = _split_patterns(exclude)
    candidates = [
        name
        for name, module in root.named_modules()
        if isinstance(module, nn.Linear)
        and not isinstance(module, LoRALinear)
        and _matches(name, include_patterns)
        and not _matches(name, exclude_patterns)
    ]
    replaced = []
    for name in candidates:
        parent, attr = _resolve_parent(root, name)
        base = getattr(parent, attr)
        if not isinstance(base, nn.Linear):
            continue
        setattr(parent, attr, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)
    return replaced


def lora_parameters(root: nn.Module):
    for module in root.modules():
        if isinstance(module, LoRALinear):
            yield from module.lora_A.parameters()
            yield from module.lora_B.parameters()


def lora_state_dict(root: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, module in root.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A.weight"] = module.lora_A.weight.detach().cpu()
            state[f"{name}.lora_B.weight"] = module.lora_B.weight.detach().cpu()
    return state


def load_lora_state_dict(root: nn.Module, state: dict[str, torch.Tensor], strict: bool = False) -> dict[str, list[str]]:
    own = dict(root.named_modules())
    missing = []
    unexpected = []
    used = set()
    for key, tensor in state.items():
        if key.endswith(".lora_A.weight"):
            module_name = key[: -len(".lora_A.weight")]
            attr = "lora_A"
        elif key.endswith(".lora_B.weight"):
            module_name = key[: -len(".lora_B.weight")]
            attr = "lora_B"
        else:
            unexpected.append(key)
            continue
        module = own.get(module_name)
        if not isinstance(module, LoRALinear):
            unexpected.append(key)
            continue
        target = getattr(module, attr).weight
        if tuple(target.shape) != tuple(tensor.shape):
            unexpected.append(key)
            continue
        target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
        used.add(key)
    if strict:
        for name, module in own.items():
            if isinstance(module, LoRALinear):
                for suffix in ("lora_A.weight", "lora_B.weight"):
                    key = f"{name}.{suffix}"
                    if key not in used:
                        missing.append(key)
    return {"missing": missing, "unexpected": unexpected}


def set_lora_train_mode(root: nn.Module, train: bool) -> None:
    for module in root.modules():
        if isinstance(module, LoRALinear):
            module.train(train)
