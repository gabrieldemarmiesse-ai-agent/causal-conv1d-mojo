"""JIT-on-first-use dispatcher for causal_conv1d bwd_full.

Each unique runtime config (dtype × n_elts × width × has_bias ×
has_seq_idx × has_initial_states × apply_silu × contig_inner ×
aligned_seq × deterministic × channel_last × use_external_stream) compiles the static
``bwd_full/variant.mojo`` once via ``mojo build -D KEY=VALUE …`` and
caches the resulting ``.so`` on disk.

Performance note (AMD-specific): The Mojo `DeviceContext()`
constructor issues `hipStreamCreate` and the matching `__del__`
issues `hipStreamDestroy`. At small-batch shapes the bwd kernel is
only ~6-10 us of GPU work, so per-call stream churn shows up in
torch.profiler. Each variant exposes a
``causal_conv1d_bwd_full_acquire_ctx`` entry point; the first call
from Python invokes it to obtain a process-lifetime DeviceContext
handle (refcount-retained so the wrapper destructor is a no-op), and
caches it. Subsequent dispatches pass that handle in to
`launch_bwd_full`, which wraps it via the doc-hidden non-owning
constructor — no new hipStream per call.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from causal_conv1d_mojo._jit_common import compile_and_load, detect_gpu_backend

_BWD_DIR = Path(__file__).resolve().parent
_PKG_DIR = _BWD_DIR.parent
_VARIANT_MOJO = _BWD_DIR / "variant.mojo"

_DTYPE_NAME = {0: "fp16", 1: "bf16", 2: "fp32"}
_DTYPE_DEFINE = {0: "float16", 1: "bfloat16", 2: "float32"}

# Wide per-thread element count (16-byte LDG): 8 for fp16/bf16, 4 for
# fp32. Mirrors `kNEltsBwd_for` in bwd_full/common.mojo.
_KN_ELTS_WIDE = {0: 8, 1: 8, 2: 4}
_KN_ELTS_NARROW = 4
# Block size (`kNThreads` in bwd_full/common.mojo).
_KNTHREADS = 128
_ELEMENT_SIZE = {0: 2, 1: 2, 2: 4}


def _rows_are_16b_aligned(
    ptr: int, batch_stride: int, channel_stride: int, element_size: int
) -> bool:
    """Whether every (batch, channel) row starts on a 16-byte boundary."""
    return (
        ptr % 16 == 0
        and (batch_stride * element_size) % 16 == 0
        and (channel_stride * element_size) % 16 == 0
    )


def call_bwd_full(args: tuple, pre_dispatch: Callable[[], None] | None = None) -> None:
    """JIT-compile (if needed) and dispatch a single bwd_full call.

    ``pre_dispatch`` runs after the (possibly seconds-long) JIT compile
    and immediately before the launch — the MPS path uses it to revive
    the argument tensors' MTLHeaps inside the driver's ~1 s residency
    window (see ``_mps.revive_heaps``).
    """
    variant_fn, ctx_handle = _get_variant_fn(_config_from_args(args))
    if pre_dispatch is not None:
        pre_dispatch()
    # Tack ctx_handle on as the 42nd positional arg — the variant
    # entry point destructures `args[41]` for it. (args[39:41] are the
    # `use_external_stream` and `deterministic` flags, already comptime
    # defines.)
    variant_fn(*args, ctx_handle)


def _config_from_args(args: tuple) -> tuple:
    dtype_code = args[23]
    width = args[25]
    has_bias = bool(args[21])
    apply_silu = bool(args[22])
    has_seq_idx = bool(args[26])
    has_initial_states = bool(args[30])
    seqlen = args[9]
    contig_inner = (
        args[12] == 1  # x_l_stride
        and args[14] == 1  # w_w_stride
        and args[17] == 1  # dout_l_stride
        and args[20] == 1  # dx_l_stride
    )
    element_size = _ELEMENT_SIZE[dtype_code]
    rows_16b_aligned = (
        _rows_are_16b_aligned(args[0], args[10], args[11], element_size)
        and _rows_are_16b_aligned(args[3], args[15], args[16], element_size)
        and _rows_are_16b_aligned(args[4], args[18], args[19], element_size)
    )

    n_elts_wide = _KN_ELTS_WIDE[dtype_code]
    deterministic = bool(args[40])
    # A `(B,L,D)`-contiguous allocation transposed to `(B,D,L)` has
    # channel stride one and seqlen stride >1. The dedicated kernel
    # vectorizes x/dout/dx along channels, so every base and row/batch
    # stride participating in a 16-byte access must preserve alignment.
    # Width >5 remains on the generic kernel (fp16/bf16 generic supports
    # up to 9); the channel-last register arrays are sized for W<=5.
    #
    # Deterministic mode is the one hard exclusion: the channel-last
    # kernel reduces dweight/dbias with device-scope float atomics, which
    # is exactly the across-block nondeterminism `deterministic` exists to
    # remove. Fall back to the (workspace-capable) generic kernel there
    # until a `(B, n_L_chunks, D, W)` workspace variant is added here.
    channel_last = (
        not deterministic
        and width <= 5
        and args[11] == 1  # x_c_stride
        and args[12] > 1  # x_l_stride
        and args[14] == 1  # weight width-contiguous
        and args[16] == 1  # dout_c_stride
        and args[17] > 1  # dout_l_stride
        and args[19] == 1  # dx_c_stride
        and args[20] > 1  # dx_l_stride
        and args[8] % n_elts_wide == 0  # dim
        and args[10] % n_elts_wide == 0  # x batch stride
        and args[12] % n_elts_wide == 0  # x seqlen stride
        and args[15] % n_elts_wide == 0  # dout batch stride
        and args[17] % n_elts_wide == 0  # dout seqlen stride
        and args[18] % n_elts_wide == 0  # dx batch stride
        and args[20] % n_elts_wide == 0  # dx seqlen stride
        and args[0] % 16 == 0  # x base
        and args[3] % 16 == 0  # dout base
        and args[4] % 16 == 0  # dx base
        and (not has_bias or args[2] % 16 == 0)  # bias base
    )
    if channel_last:
        # Generic-only choices are pinned so seqlen/layout details do not
        # fragment the channel-last cache. The CL kernel derives its own
        # 16-byte vector width from dtype and handles tail rows directly.
        n_elts = n_elts_wide
        contig_inner = False
        aligned_seq = False
    else:
        # Match the original dispatcher's runtime n_elts pick: wide only if
        # (a) wide differs from narrow (i.e. dtype is 16-bit), (b) seqlen is
        # aligned to kNThreads * wide, and (c) every row of every tensor this
        # variant vector-accesses (x, dout, dx) is 16-byte aligned. Otherwise
        # use the narrow, alignment-agnostic variant.
        #
        # (c) applies only to the contiguous-inner path: that is the one that
        # issues 16-byte accesses along a row. A strided x (any non-unit
        # seqlen stride) is read scalar, so its row bases can't fault and the
        # wide halo stays available — which is what widths 6..9 need.
        vec_rows_ok = rows_16b_aligned or not contig_inner
        use_wide = (
            vec_rows_ok
            and n_elts_wide != _KN_ELTS_NARROW
            and (seqlen % (_KNTHREADS * n_elts_wide)) == 0
        )
        n_elts = n_elts_wide if use_wide else _KN_ELTS_NARROW
        aligned_seq = vec_rows_ok and (seqlen % (_KNTHREADS * n_elts)) == 0
    # See fwd/_jit.py for why this is comptime instead of a runtime
    # branch on `stream_handle_addr`. Python wrapper sets 1 for CUDA,
    # 0 for Metal.
    use_external_stream = bool(args[39])

    return (
        dtype_code,
        n_elts,
        width,
        has_bias,
        has_seq_idx,
        has_initial_states,
        apply_silu,
        contig_inner,
        aligned_seq,
        deterministic,
        channel_last,
        use_external_stream,
    )


def _mod_name(config: tuple) -> str:
    (dt, ne, w, hb, hs, hi, silu, c, a, det, cl, ues) = config
    return (
        f"{_DTYPE_NAME[dt]}_n{ne}_w{w}"
        f"_hb{int(hb)}_hs{int(hs)}_hi{int(hi)}_silu{int(silu)}"
        f"_contig{int(c)}_chunk16{int(a)}_det{int(det)}_cl{int(cl)}"
        f"_extstr{int(ues)}"
    )


def _defines(config: tuple) -> dict[str, str]:
    (dt, ne, w, hb, hs, hi, silu, c, a, det, cl, ues) = config

    def b(x: bool) -> str:
        return "true" if x else "false"

    return {
        "DTYPE": _DTYPE_DEFINE[dt],
        "N_ELTS": str(ne),
        "WIDTH": str(w),
        "HAS_BIAS": b(hb),
        "HAS_SEQ_IDX": b(hs),
        "HAS_INITIAL_STATES": b(hi),
        "APPLY_SILU": b(silu),
        "CONTIG_INNER": b(c),
        "ALIGNED_SEQ": b(a),
        "DETERMINISTIC": b(det),
        "CHANNEL_LAST": b(cl),
        "USE_EXTERNAL_STREAM": b(ues),
    }


@lru_cache(maxsize=None)
def _get_variant_fn(config: tuple) -> tuple[Callable, int]:
    mod_name = _mod_name(config)
    backend, backend_arch = detect_gpu_backend()
    module = compile_and_load(
        subpkg="bwd_full",
        source_file=_VARIANT_MOJO,
        include_dirs=(_BWD_DIR, _PKG_DIR),
        defines=_defines(config),
        mod_name=mod_name,
        backend=backend,
        backend_arch=backend_arch,
    )
    fn = module.causal_conv1d_bwd_full_variant
    acquire = module.causal_conv1d_bwd_full_acquire_ctx
    ctx_handle = int(acquire(()))
    return fn, ctx_handle
