"""Packed-sequence coverage with initial states (upstream v1.7.0 parity).

The initial-state values occupy virtual positions before t=0 and carry
``seq_idx[b, 0]``. They therefore contribute only to the first packed
sequence in each batch row. These tests cover the reference oracle plus
forward and all differentiable inputs of the CPU/CUDA kernels.
"""

from __future__ import annotations

import pytest
import torch

from causal_conv1d_mojo import causal_conv1d_fn, causal_conv1d_ref

from _helpers import _DX_TOL, _FWD_TOL, _assert_dw_close, _max_diff


def _per_segment_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_states: torch.Tensor,
    seq_idx: torch.Tensor,
    activation: str | None,
) -> torch.Tensor:
    """Independent oracle: run each packed fragment as its own conv.

    Only a valid fragment beginning at t=0 receives initial_states.
    Padding fragments return zero, and a later valid fragment never
    inherits the state even when the row begins with padding.
    """
    batch, _, seqlen = x.shape
    rows = []
    for b in range(batch):
        ids = seq_idx[b].tolist()
        fragments = []
        start = 0
        while start < seqlen:
            end = start + 1
            while end < seqlen and ids[end] == ids[start]:
                end += 1
            if ids[start] < 0:
                fragments.append(torch.zeros_like(x[b : b + 1, :, start:end]))
            else:
                fragments.append(
                    causal_conv1d_ref(
                        x[b : b + 1, :, start:end],
                        weight,
                        bias,
                        initial_states=initial_states[b : b + 1]
                        if start == 0
                        else None,
                        activation=activation,
                    )
                )
            start = end
        rows.append(torch.cat(fragments, dim=-1))
    return torch.cat(rows, dim=0)


def _packed_seq_idx(
    seqlen: int, width: int, device: str
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Three valid fragments plus padding, with row-specific first lengths."""
    first_lengths = (max(1, width - 2), min(width, seqlen - 3))
    rows = []
    for b, first_len in enumerate(first_lengths):
        remaining = seqlen - first_len
        second_len = max(1, min(width + 1, remaining - 2))
        third_len = remaining - second_len - 1
        rows.append(
            torch.cat(
                [
                    torch.full(
                        (first_len,), 3 * b + 3, dtype=torch.int32, device=device
                    ),
                    torch.full(
                        (second_len,), 3 * b + 4, dtype=torch.int32, device=device
                    ),
                    torch.full((1,), -1, dtype=torch.int32, device=device),
                    torch.full(
                        (third_len,), 3 * b + 5, dtype=torch.int32, device=device
                    ),
                ]
            )
        )
    return torch.stack(rows), first_lengths


def test_causal_conv1d_ref_varlen_initial_states(device):
    """Upstream's reference test: packed ref equals per-fragment calls."""
    batch, dim, seqlen, width = 2, 8, 7, 4
    x = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    initial_states = torch.randn(
        batch, dim, width - 1, device=device, dtype=torch.float32
    )
    seq_idx = torch.tensor(
        [[3, 5, 5, 5, 9, 9, 9], [4, 4, 6, 6, 6, 6, 8]],
        device=device,
        dtype=torch.int32,
    )

    out = causal_conv1d_ref(
        x,
        weight,
        bias,
        initial_states=initial_states,
        activation="silu",
        seq_idx=seq_idx,
    )
    out_ref = _per_segment_reference(x, weight, bias, initial_states, seq_idx, "silu")
    torch.testing.assert_close(out, out_ref, rtol=3e-4, atol=1e-3)


# Every supported width gets a dual-flag forward and backward JIT on each
# available device. The cases also deliberately span aligned/unaligned
# seqlens, contiguous/channel-last x, bias/no-bias, and silu/no activation.
_VARLEN_CASES = [
    pytest.param(torch.float16, 2, 8, False, True, None, id="fp16-w2-contig"),
    pytest.param(torch.float16, 2, 151, True, False, "silu", id="fp16-w2-cl"),
    pytest.param(torch.float16, 3, 151, True, False, "silu", id="fp16-w3-cl"),
    pytest.param(torch.float16, 4, 8, False, False, "silu", id="fp16-w4-contig"),
    pytest.param(torch.float16, 4, 151, True, True, None, id="fp16-w4-cl"),
    pytest.param(torch.float16, 5, 151, True, True, None, id="fp16-w5-cl"),
    pytest.param(torch.float16, 6, 1024, False, False, "silu", id="fp16-w6"),
    pytest.param(torch.float16, 7, 1024, True, True, None, id="fp16-w7-cl"),
    pytest.param(torch.float16, 8, 1024, False, True, "silu", id="fp16-w8"),
    pytest.param(torch.float16, 9, 1024, True, False, "silu", id="fp16-w9-cl"),
    pytest.param(torch.bfloat16, 2, 151, True, False, None, id="bf16-w2-cl"),
    pytest.param(torch.bfloat16, 3, 8, False, True, "silu", id="bf16-w3"),
    pytest.param(torch.bfloat16, 3, 151, True, False, "silu", id="bf16-w3-cl"),
    pytest.param(torch.bfloat16, 4, 151, True, True, "silu", id="bf16-w4-cl"),
    pytest.param(torch.bfloat16, 5, 8, False, False, None, id="bf16-w5"),
    pytest.param(torch.bfloat16, 5, 151, True, True, None, id="bf16-w5-cl"),
    pytest.param(torch.bfloat16, 6, 1024, True, True, "silu", id="bf16-w6-cl"),
    pytest.param(torch.bfloat16, 7, 1024, False, False, None, id="bf16-w7"),
    pytest.param(torch.bfloat16, 8, 1024, True, False, "silu", id="bf16-w8-cl"),
    pytest.param(torch.bfloat16, 9, 1024, False, True, "silu", id="bf16-w9"),
    pytest.param(torch.float32, 2, 8, False, False, "silu", id="fp32-w2"),
    pytest.param(torch.float32, 2, 151, True, True, None, id="fp32-w2-cl"),
    pytest.param(torch.float32, 3, 151, True, True, None, id="fp32-w3-cl"),
    pytest.param(torch.float32, 4, 8, False, True, "silu", id="fp32-w4"),
    pytest.param(torch.float32, 4, 151, True, False, "silu", id="fp32-w4-cl"),
    pytest.param(torch.float32, 5, 151, True, False, "silu", id="fp32-w5-cl"),
]


@pytest.mark.parametrize(
    "dtype,width,seqlen,channel_last,bias_present,activation", _VARLEN_CASES
)
def test_causal_conv1d_varlen_initial_states(
    device,
    dtype,
    width,
    seqlen,
    channel_last,
    bias_present,
    activation,
):
    """Forward and dx/dweight/dbias/dinitial_states match the v1.7 ref."""
    batch, dim = 2, 16
    if channel_last:
        x = (
            torch.randn(batch, seqlen, dim + 16, device=device, dtype=dtype)[:, :, :dim]
            .transpose(1, 2)
            .requires_grad_()
        )
        assert x.stride(1) == 1 and x.stride(2) != 1
    else:
        x = torch.randn(
            batch, dim, seqlen, device=device, dtype=dtype, requires_grad=True
        )
        assert x.is_contiguous()

    weight = torch.randn(dim, width, device=device, dtype=dtype, requires_grad=True)
    bias = (
        torch.randn(dim, device=device, dtype=dtype, requires_grad=True)
        if bias_present
        else None
    )
    initial_states = (
        torch.randn(batch, width - 1, dim, device=device, dtype=dtype)
        .transpose(1, 2)
        .requires_grad_()
    )
    seq_idx, first_lengths = _packed_seq_idx(seqlen, width, device)
    if width > 2:
        assert first_lengths[0] < width - 1
    assert first_lengths[0] != first_lengths[1]

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    initial_states_ref = initial_states.detach().clone().requires_grad_()

    out = causal_conv1d_fn(
        x,
        weight,
        bias,
        seq_idx=seq_idx,
        initial_states=initial_states,
        activation=activation,
    )
    out_ref = causal_conv1d_ref(
        x_ref,
        weight_ref,
        bias_ref,
        initial_states=initial_states_ref,
        activation=activation,
        seq_idx=seq_idx,
    )

    assert _max_diff(out, out_ref) < _FWD_TOL[dtype]
    padding = (seq_idx < 0).unsqueeze(1).expand_as(out)
    assert torch.count_nonzero(out.masked_select(padding)) == 0

    dout = torch.randn_like(out)
    out.backward(dout)
    out_ref.backward(dout)

    assert _max_diff(x.grad, x_ref.grad) < _DX_TOL[dtype]
    _assert_dw_close(weight.grad, weight_ref.grad, dtype, name="dweight")
    if bias is not None:
        _assert_dw_close(bias.grad, bias_ref.grad, dtype, name="dbias")
    assert _max_diff(initial_states.grad, initial_states_ref.grad) < _DX_TOL[dtype]
