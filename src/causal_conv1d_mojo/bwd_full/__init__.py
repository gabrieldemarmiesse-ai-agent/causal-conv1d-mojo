"""GPU fused backward subpackage: kernel + JIT dispatcher + Python wrapper."""

from __future__ import annotations

import torch

from causal_conv1d_mojo._dtype import _DTYPE_CODE, _ptr
from causal_conv1d_mojo._mps import gpu_address, gpu_address_or_zero, revive_heaps


def native_bwd_full(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    dx: torch.Tensor,
    dweight_acc: torch.Tensor,
    dbias_acc: torch.Tensor | None,
    dinitial_states: torch.Tensor | None,
    apply_silu: bool,
    deterministic: bool,
) -> None:
    # 41-tuple expected by the JIT-generated variant entry point.
    # Each unique runtime config lazily compiles its own single-variant
    # `.so` on first use, then caches it under
    # `$XDG_CACHE_HOME/causal_conv1d_mojo/bwd_full/`.
    from causal_conv1d_mojo.bwd_full._jit import call_bwd_full

    call_bwd_full(
        (
            x.data_ptr(),
            weight.data_ptr(),
            _ptr(bias),
            dout.data_ptr(),
            dx.data_ptr(),
            dweight_acc.data_ptr(),
            _ptr(dbias_acc),
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.stride(0),
            x.stride(1),
            x.stride(2),
            weight.stride(0),
            weight.stride(1),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dx.stride(0),
            dx.stride(1),
            dx.stride(2),
            int(bias is not None),
            int(apply_silu),
            _DTYPE_CODE[x.dtype],
            _DTYPE_CODE[weight.dtype],
            torch.cuda.current_stream().cuda_stream,
            weight.shape[1],
            int(seq_idx is not None),
            _ptr(seq_idx),
            seq_idx.stride(0) if seq_idx is not None else 0,
            seq_idx.stride(1) if seq_idx is not None else 0,
            int(initial_states is not None),
            _ptr(initial_states),
            initial_states.stride(0) if initial_states is not None else 0,
            initial_states.stride(1) if initial_states is not None else 0,
            initial_states.stride(2) if initial_states is not None else 0,
            _ptr(dinitial_states),
            dinitial_states.stride(0) if dinitial_states is not None else 0,
            dinitial_states.stride(1) if dinitial_states is not None else 0,
            dinitial_states.stride(2) if dinitial_states is not None else 0,
            1,  # use_external_stream: CUDA path wraps torch's stream
            int(deterministic),
        )
    )


def native_bwd_full_mps(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dout: torch.Tensor,
    seq_idx: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    dx: torch.Tensor,
    dweight_acc: torch.Tensor,
    dbias_acc: torch.Tensor | None,
    dinitial_states: torch.Tensor | None,
    apply_silu: bool,
    deterministic: bool,
) -> None:
    """Mac/MPS path — see `fwd/__init__.py::native_fwd_mps` for the
    rationale (torch MPS data_ptr is an Obj-C MTLBuffer pointer; we
    extract Metal `gpuAddress` instead, and pass `stream_handle=0`).
    """
    from causal_conv1d_mojo.bwd_full._jit import call_bwd_full

    # Same 40-tuple ABI as CUDA. `_jit.py` derives channel-last dispatch
    # from these addresses/strides, so the optimized path adds no ABI slot.

    def _pre_dispatch() -> None:
        # Post-compile, pre-launch: revive every argument tensor's
        # MTLHeap (see _mps.revive_heaps), then flush torch's queue.
        revive_heaps(
            x,
            weight,
            bias,
            dout,
            seq_idx,
            initial_states,
            dx,
            dweight_acc,
            dbias_acc,
            dinitial_states,
        )
        torch.mps.synchronize()

    call_bwd_full(
        (
            gpu_address(x),
            gpu_address(weight),
            gpu_address_or_zero(bias),
            gpu_address(dout),
            gpu_address(dx),
            gpu_address(dweight_acc),
            gpu_address_or_zero(dbias_acc),
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.stride(0),
            x.stride(1),
            x.stride(2),
            weight.stride(0),
            weight.stride(1),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dx.stride(0),
            dx.stride(1),
            dx.stride(2),
            int(bias is not None),
            int(apply_silu),
            _DTYPE_CODE[x.dtype],
            _DTYPE_CODE[weight.dtype],
            0,  # stream_handle_addr — Metal has no streams
            weight.shape[1],
            int(seq_idx is not None),
            gpu_address_or_zero(seq_idx),
            seq_idx.stride(0) if seq_idx is not None else 0,
            seq_idx.stride(1) if seq_idx is not None else 0,
            int(initial_states is not None),
            gpu_address_or_zero(initial_states),
            initial_states.stride(0) if initial_states is not None else 0,
            initial_states.stride(1) if initial_states is not None else 0,
            initial_states.stride(2) if initial_states is not None else 0,
            gpu_address_or_zero(dinitial_states),
            dinitial_states.stride(0) if dinitial_states is not None else 0,
            dinitial_states.stride(1) if dinitial_states is not None else 0,
            dinitial_states.stride(2) if dinitial_states is not None else 0,
            0,  # use_external_stream: Metal path enqueues on ctx
            int(deterministic),
        ),
        pre_dispatch=_pre_dispatch,
    )
