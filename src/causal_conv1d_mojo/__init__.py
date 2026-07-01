"""causal_conv1d_update, fused into a Mojo kernel and called via a direct
Python <-> Mojo CPython extension (no MAX framework). Trimmed repro of a
Mojo/Metal cold-cache bug — see repro.py.

torch MPS `tensor.data_ptr()` is an Obj-C `MTLBuffer` object pointer, not
the GPU virtual address Mojo needs — `gpu_address()` below extracts the
real `gpuAddress` via the same selector Mojo uses internally:

    [storage.MTLBuffer gpuAddress] + (tensor.data_ptr() - storage.data_ptr())
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.machinery
import importlib.util
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import torch

from mojo.run import subprocess_run_mojo

__all__ = ["causal_conv1d_update"]

_PKG_DIR = Path(__file__).resolve().parent
_VARIANT_MOJO = _PKG_DIR / "variant.mojo"


@lru_cache(maxsize=1)
def _objc() -> ctypes.CDLL:
    libobjc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    libobjc.sel_registerName.restype = ctypes.c_void_p
    libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
    libobjc.objc_msgSend.restype = ctypes.c_uint64
    libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return libobjc


def gpu_address(t: torch.Tensor) -> int:
    storage = t.untyped_storage()
    buf_obj = storage.data_ptr()
    if buf_obj == 0:
        return 0
    libobjc = _objc()
    sel = libobjc.sel_registerName(b"gpuAddress")
    base_gpu = libobjc.objc_msgSend(buf_obj, sel)
    return base_gpu + (t.data_ptr() - buf_obj)


@lru_cache(maxsize=None)
def _get_variant_fn():
    so_path = Path(tempfile.mkstemp(suffix=".so")[1])
    cmd = ["build", str(_VARIANT_MOJO), "--emit", "shared-lib", "-o", str(so_path)]
    print("[causal_conv1d_mojo] compiling update kernel...", file=sys.stderr, end="")
    subprocess_run_mojo(cmd, check=True)
    print(" done", file=sys.stderr)

    loader = importlib.machinery.ExtensionFileLoader("variant", str(so_path))
    spec = importlib.util.spec_from_loader("variant", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    return module.causal_conv1d_update_variant


def causal_conv1d_update(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)

    torch.mps.synchronize()
    variant_fn = _get_variant_fn()
    variant_fn(
        gpu_address(x),
        gpu_address(out),
    )

    return out
