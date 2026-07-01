"""JIT-on-first-use dispatcher for causal_conv1d update (trimmed repro)."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from causal_conv1d_mojo._jit_common import compile_and_load, detect_gpu_backend

_UPDATE_DIR = Path(__file__).resolve().parent
_PKG_DIR = _UPDATE_DIR.parent
_VARIANT_MOJO = _UPDATE_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}


def call_update(args: tuple) -> None:
    config = _config_from_args(args)
    variant_fn, ctx_handle = _get_variant_fn(config)
    variant_fn(*args, ctx_handle)


def _config_from_args(args: tuple) -> tuple:
    return (
        args[21],  # dtype_code
        args[22],  # width
        bool(args[19]),  # has_bias
        bool(args[20]),  # apply_silu
    )


def _mod_name(config: tuple) -> str:
    (dt, w, hb, silu) = config
    return f"{_DTYPE_NAME[dt]}_w{w}_hb{int(hb)}_silu{int(silu)}"


def _defines(config: tuple) -> dict[str, str]:
    (dt, w, hb, silu) = config

    def b(x: bool) -> str:
        return "true" if x else "false"

    return {
        "DTYPE": _DTYPE_DEFINE[dt],
        "WIDTH": str(w),
        "HAS_BIAS": b(hb),
        "APPLY_SILU": b(silu),
    }


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple) -> tuple[Callable, int]:
    mod_name = _mod_name(config)
    backend, backend_arch = detect_gpu_backend()
    module = compile_and_load(
        subpkg="update",
        source_file=_VARIANT_MOJO,
        include_dirs=(_UPDATE_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=mod_name,
        backend=backend,
        backend_arch=backend_arch,
    )
    fn = module.causal_conv1d_update_variant
    acquire = module.causal_conv1d_update_acquire_ctx
    ctx_handle = int(acquire(()))
    return fn, ctx_handle
