"""GPU single-step update subpackage: kernel + JIT dispatcher + Python wrapper."""

from __future__ import annotations

import torch

from causal_conv1d_mojo._mps import gpu_address


def native_update_mps(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    conv_state: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """torch MPS data_ptr is an Obj-C MTLBuffer pointer; extract Metal
    `gpuAddress` instead (see _mps.py)."""
    from causal_conv1d_mojo.update._jit import call_update

    torch.mps.synchronize()
    call_update(
        (
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
        )
    )
