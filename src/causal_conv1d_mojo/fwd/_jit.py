"""JIT-on-first-use dispatcher for causal_conv1d fwd.

Each unique runtime config (dtype × width × has_bias × has_seq_idx ×
has_initial_states × apply_silu × contig_inner × aligned_seq ×
vec_aligned × use_external_stream) compiles the static
``fwd/variant.mojo`` once via ``mojo build -D KEY=VALUE …``, caches
the resulting ``.so`` on disk, and dispatches into it on every call.
The first call per (config, machine) pays the compile cost; every
later call in this or any future process hits the cache.

The compile + cache + load plumbing lives in
``causal_conv1d_mojo._jit_common.compile_and_load``. This module
owns the fwd-specific bits: how to read the config out of the
Python-side args, how to name a variant (cache key), and how to
materialise the config as a `-D` defines mapping.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from causal_conv1d_mojo._jit_common import compile_and_load, detect_gpu_backend

_FWD_DIR = Path(__file__).resolve().parent
_PKG_DIR = _FWD_DIR.parent
_VARIANT_MOJO = _FWD_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
# `get_defined_dtype` in std.sys parses these via `DType._from_str`.
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}

# Per-thread element count: 8 for fp16/bf16, 4 for fp32. Mirrors
# `kNEltsFwd` in fwd/common.mojo.
_KN_ELTS = {0: 8, 1: 8, 2: 4}
# Block size (`kNThreads` in fwd/common.mojo).
_KNTHREADS = 128


def call_fwd(args: tuple, pre_dispatch: Callable[[], None] | None = None) -> None:
    """JIT-compile (if needed) and dispatch a single fwd call.

    ``args`` is the 29-tuple of runtime values built by
    ``fwd/__init__.py::native_fwd``. ``pre_dispatch`` runs after the
    (possibly seconds-long) JIT compile and immediately before the
    kernel launch — the MPS path uses it to revive the argument
    tensors' MTLHeaps inside the driver's ~1 s residency window (see
    ``_mps.revive_heaps``); doing it any earlier would let a slow
    compile re-idle the heaps.
    """
    variant_fn, ctx_handle = _get_variant_fn(_config_from_args(args))
    if pre_dispatch is not None:
        pre_dispatch()
    # Tack ctx_handle on as the 30th positional arg — the variant
    # entry point destructures `args[29]` for it. Avoids the per-call
    # hipStreamCreate/Destroy churn from `var ctx = DeviceContext()`.
    variant_fn(*args, ctx_handle)


def _config_from_args(args: tuple) -> tuple:
    dtype_code = args[17]
    width = args[23]
    has_bias = bool(args[15])
    apply_silu = bool(args[16])
    has_seq_idx = bool(args[19])
    has_initial_states = bool(args[24])
    seqlen = args[6]
    contig_inner = args[9] == 1 and args[11] == 1 and args[14] == 1
    aligned_seq = (seqlen % (_KNTHREADS * _KN_ELTS[dtype_code])) == 0
    # `vec_aligned` is the weaker "seqlen % kNElts == 0" — true for any
    # power-of-two seqlen on fp16/bf16 (kNElts=8) and fp32 (kNElts=4).
    # When this holds, every thread's kNElts slice either fits entirely
    # inside [0, seqlen) or starts past it, so the partial-chunk scalar
    # fallback path in the kernel becomes statically dead. Mirrors
    # upstream's `kIsVecLoad` BOOL_SWITCH (gated on the same condition).
    vec_aligned = (seqlen % _KN_ELTS[dtype_code]) == 0
    # `use_external_stream` is a comptime gate: True for CUDA/HIP (wrap
    # torch's CUstream/hipStream and enqueue on it), False for Metal
    # (no CUDA-style streams; enqueue on the DeviceContext directly +
    # sync after). Passed as comptime so the variant only codegens one
    # branch — a runtime `if` here costs ~30 μs/call on NVIDIA, even
    # when the branch is perfectly predictable. `args[29]` is set by
    # the Python wrappers (`native_fwd` passes 1, `native_fwd_mps`
    # passes 0). Can't be derived from `stream_handle_addr` itself
    # because torch's default CUDA stream has cuda_stream == 0.
    use_external_stream = bool(args[29])
    # Channel-last fast path (Metal only for now — correct but untested
    # on NVIDIA/AMD hardware, where the generic strided path keeps
    # handling this layout): x/out have dim contiguous (stride(1)==1)
    # and seqlen strided — the layout a (B, L, D)-contiguous activation
    # gets after .transpose(1, 2), i.e. upstream's `is_channel_last`
    # condition. The dedicated kernel vectorizes along dim, so dim and
    # the non-contiguous strides of x/out must all be multiples of
    # kNElts to keep every 16-byte access aligned. seq_idx stays on the
    # generic path (v1 scope; mirrors it being the rarer case here —
    # note upstream is the opposite: their seq_idx REQUIRES
    # channel-last).
    kn = _KN_ELTS[dtype_code]
    channel_last = (
        not use_external_stream
        and args[8] == 1  # x dim-contiguous
        and args[9] > 1  # x seqlen strided (else the contig path wins)
        and args[13] == 1  # out dim-contiguous
        and args[14] > 1
        and args[11] == 1  # weight width-contiguous
        and not has_seq_idx
        and args[5] % kn == 0  # dim
        and args[7] % kn == 0  # x batch stride
        and args[9] % kn == 0  # x seqlen stride
        and args[12] % kn == 0  # out batch stride
        and args[14] % kn == 0  # out seqlen stride
    )
    if channel_last:
        # These three only parameterize the generic kernel; pin them so
        # every channel-last shape shares one cached variant.
        contig_inner = False
        aligned_seq = False
        vec_aligned = False
    return (
        dtype_code,
        width,
        has_bias,
        has_seq_idx,
        has_initial_states,
        apply_silu,
        contig_inner,
        aligned_seq,
        vec_aligned,
        channel_last,
        use_external_stream,
    )


def _mod_name(config: tuple) -> str:
    """Readable, deterministic identifier for a config.

    Used as the cache key. Reading it should be enough to reproduce
    the config by hand.
    """
    (dt, w, hb, hs, hi, silu, c, a, va, cl, ues) = config
    return (
        f"{_DTYPE_NAME[dt]}_w{w}"
        f"_hb{int(hb)}_hs{int(hs)}_hi{int(hi)}_silu{int(silu)}"
        f"_contig{int(c)}_aligned{int(a)}_vec{int(va)}_cl{int(cl)}"
        f"_extstr{int(ues)}"
    )


def _defines(config: tuple) -> dict[str, str]:
    """Materialise the config as `-D KEY=VALUE` pairs for `mojo build`.

    The corresponding `comptime` reads live in `fwd/variant.mojo`.
    """
    (dt, w, hb, hs, hi, silu, c, a, va, cl, ues) = config

    def b(x: bool) -> str:
        return "true" if x else "false"

    return {
        "DTYPE": _DTYPE_DEFINE[dt],
        "WIDTH": str(w),
        "HAS_BIAS": b(hb),
        "HAS_SEQ_IDX": b(hs),
        "HAS_INITIAL_STATES": b(hi),
        "APPLY_SILU": b(silu),
        "CONTIG_INNER": b(c),
        "ALIGNED_SEQ": b(a),
        "VEC_ALIGNED": b(va),
        "CHANNEL_LAST": b(cl),
        "USE_EXTERNAL_STREAM": b(ues),
    }


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple) -> tuple[Callable, int]:
    mod_name = _mod_name(config)
    backend, backend_arch = detect_gpu_backend()
    module = compile_and_load(
        subpkg="fwd",
        source_file=_VARIANT_MOJO,
        include_dirs=(_FWD_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=mod_name,
        backend=backend,
        backend_arch=backend_arch,
    )
    fn = module.causal_conv1d_fwd_variant
    acquire = module.causal_conv1d_fwd_acquire_ctx
    ctx_handle = int(acquire(()))
    return fn, ctx_handle
