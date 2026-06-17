"""Shared model/device helpers. Reused by eval (M2) and training (M4-M5).

Device policy: cuda -> mps -> cpu. dtype policy ("auto"): bf16 on cuda,
fp16 on mps (8GB-friendly), fp32 on cpu. All overridable via config.
"""

from __future__ import annotations

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DTYPE_MAP = {
    "auto": None,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(pref: str = "auto") -> torch.device:
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    dt = _DTYPE_MAP.get(str(name).lower(), None)
    if dt is not None:
        return dt
    # "auto"
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_tokenizer(model_id: str):
    """Tokenizer configured for left-padded batched generation (decoder-only)."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model_and_tokenizer(model_id: str, *, dtype_name: str = "auto", device=None):
    device = device or get_device()
    dtype = resolve_dtype(dtype_name, device)
    tok = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tok, device
