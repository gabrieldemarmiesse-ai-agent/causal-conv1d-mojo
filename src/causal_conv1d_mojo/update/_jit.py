"""JIT-on-first-use dispatcher for causal_conv1d update (trimmed repro:
single fixed fp16/width-4 variant, no -D defines needed)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from causal_conv1d_mojo._jit_common import compile_and_load, detect_gpu_backend

_UPDATE_DIR = Path(__file__).resolve().parent
_PKG_DIR = _UPDATE_DIR.parent
_VARIANT_MOJO = _UPDATE_DIR / "variant.mojo"


def call_update(args: tuple) -> None:
    variant_fn, ctx_handle = _get_variant_fn()
    variant_fn(*args, ctx_handle)


@lru_cache(maxsize=None)
def _get_variant_fn():
    backend, backend_arch = detect_gpu_backend()
    module = compile_and_load(
        subpkg="update",
        source_file=_VARIANT_MOJO,
        include_dirs=(_UPDATE_DIR, _PKG_DIR),
        mod_name="fp16_w4",
        backend=backend,
        backend_arch=backend_arch,
    )
    fn = module.causal_conv1d_update_variant
    acquire = module.causal_conv1d_update_acquire_ctx
    ctx_handle = int(acquire(()))
    return fn, ctx_handle
