"""JIT-on-first-use dispatcher for causal_conv1d update (trimmed repro:
single fixed fp16/width-4 variant, no caching)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from mojo.run import subprocess_run_mojo

_UPDATE_DIR = Path(__file__).resolve().parent
_PKG_DIR = _UPDATE_DIR.parent
_VARIANT_MOJO = _UPDATE_DIR / "variant.mojo"


def call_update(args: tuple) -> None:
    variant_fn, ctx_handle = _get_variant_fn()
    variant_fn(*args, ctx_handle)


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
