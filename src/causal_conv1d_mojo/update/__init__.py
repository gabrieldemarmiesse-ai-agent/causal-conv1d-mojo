"""GPU single-step update subpackage: kernel + JIT dispatcher + Python wrapper."""

from __future__ import annotations

import torch

from causal_conv1d_mojo._dtype import _DTYPE_CODE, _ptr
from causal_conv1d_mojo._mps import gpu_address, gpu_address_or_zero


def native_update_mps(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    out: torch.Tensor,
    apply_silu: bool,
) -> None:
    """torch MPS data_ptr is an Obj-C MTLBuffer pointer; extract Metal
    `gpuAddress` instead (see _mps.py)."""
    from causal_conv1d_mojo.update._jit import call_update

    torch.mps.synchronize()
    call_update(
        (
            gpu_address(x),
            gpu_address(weight),
            gpu_address_or_zero(bias),
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
            weight.stride(1),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            int(bias is not None),
            int(apply_silu),
            _DTYPE_CODE[x.dtype],
            0,  # stream_handle_addr — Metal has no streams
            weight.shape[1],
            0,  # has_state_indices (always false in this trimmed repro)
            0,  # state_indices_addr
            0,  # has_cache_seqlens (always false in this trimmed repro)
            0,  # cache_seqlens_addr
            0,  # use_external_stream: Metal path enqueues on ctx
        )
    )
