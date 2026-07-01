"""GPU single-step update subpackage: kernel + JIT dispatcher + Python
wrapper (trimmed repro: single fixed fp16/width-4 variant, no caching)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import torch

from causal_conv1d_mojo._mps import gpu_address
from mojo.run import subprocess_run_mojo

_UPDATE_DIR = Path(__file__).resolve().parent
_PKG_DIR = _UPDATE_DIR.parent
_VARIANT_MOJO = _UPDATE_DIR / "variant.mojo"


@lru_cache(maxsize=None)
def _get_variant_fn():
    so_path = Path(tempfile.mkstemp(suffix=".so")[1])
    cmd = [
        "build",
        str(_VARIANT_MOJO),
        "-I",
        str(_UPDATE_DIR),
        "-I",
        str(_PKG_DIR),
        "--emit",
        "shared-lib",
        "-o",
        str(so_path),
    ]
    print("[causal_conv1d_mojo] compiling update kernel...", file=sys.stderr, end="")
    subprocess_run_mojo(cmd, check=True)
    print(" done", file=sys.stderr)

    loader = importlib.machinery.ExtensionFileLoader("variant", str(so_path))
    spec = importlib.util.spec_from_loader("variant", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    fn = module.causal_conv1d_update_variant
    acquire = module.causal_conv1d_update_acquire_ctx
    ctx_handle = int(acquire(()))
    return fn, ctx_handle


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """torch MPS data_ptr is an Obj-C MTLBuffer pointer; extract Metal
    `gpuAddress` instead (see _mps.py)."""
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    out = torch.empty_like(x)

    torch.mps.synchronize()
    variant_fn, ctx_handle = _get_variant_fn()
    variant_fn(
        gpu_address(x),
        gpu_address(weight),
        gpu_address(bias),
        gpu_address(conv_state),
        gpu_address(out),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        conv_state.shape[2],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        ctx_handle,
    )

    if unsqueeze:
        out = out.squeeze(-1)
    return out
