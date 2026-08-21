"""Alignment regressions for generic fwd/bwd and update view handling.

The GPU kernels may issue 16-byte global accesses only when every row
base preserves that alignment. These cases deliberately use odd sequence
lengths, pointer-offset views, and outer strides that violate the vector
promise while retaining an inner stride of one.
"""

import pytest
import torch

import causal_conv1d_mojo
from causal_conv1d_mojo import causal_conv1d_ref, causal_conv1d_update_ref

from _helpers import (
    _DX_TOL,
    _FWD_TOL,
    _assert_dw_close,
    _max_diff,
    _ref_grads,
    _ref_grads_with_seq_idx,
    _ref_with_seq_idx,
)


def _assert_standard_grads(x, weight, bias, dout, activation):
    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, activation)
    assert _max_diff(x.grad, dx_ref) < _DX_TOL[x.dtype]
    _assert_dw_close(weight.grad, dw_ref, x.dtype, name="dw")
    if bias is not None:
        _assert_dw_close(bias.grad, db_ref, x.dtype, name="db")


@pytest.mark.parametrize("seqlen", [9, 17, 37, 100, 1000, 1023, 4095])
@pytest.mark.parametrize("context", ["plain", "initial_states", "seq_idx"])
@pytest.mark.parametrize("bias_present", [False, True], ids=["no_bias", "bias"])
def test_contiguous_arbitrary_seqlen_fwd_bwd(
    device, dtype, seqlen, context, bias_present
):
    """Odd row sizes exercise scalar global access for all fwd/bwd inputs."""
    B, D, W = 2, 8, 4
    x = torch.randn(B, D, seqlen, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = (
        torch.randn(D, dtype=dtype, device=device, requires_grad=True)
        if bias_present
        else None
    )
    dout = torch.randn_like(x)

    initial_states = None
    seq_idx = None
    if context == "initial_states":
        initial_states = torch.randn(
            B, D, W - 1, dtype=dtype, device=device, requires_grad=True
        )
    elif context == "seq_idx":
        first = seqlen // 3
        second = 2 * seqlen // 3
        seq_idx = torch.empty(B, seqlen, dtype=torch.int32, device=device)
        seq_idx[:, :first] = 0
        seq_idx[:, first:second] = 1
        seq_idx[:, second:] = 2

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        activation="silu",
    )

    if context == "plain":
        out_ref = causal_conv1d_ref(x, weight, bias=bias, activation="silu")
    elif context == "seq_idx":
        out_ref = _ref_with_seq_idx(x, weight, bias, seq_idx, "silu")
    else:
        out_ref = causal_conv1d_ref(
            torch.cat([initial_states, x], dim=-1),
            weight,
            bias=bias,
            activation="silu",
        )[..., W - 1 :]
    assert _max_diff(out, out_ref) < _FWD_TOL[dtype]

    out.backward(dout)
    if context == "plain":
        _assert_standard_grads(x, weight, bias, dout, "silu")
    elif context == "seq_idx":
        dx_ref, dw_ref, db_ref = _ref_grads_with_seq_idx(
            x, weight, bias, seq_idx, dout, "silu"
        )
        assert _max_diff(x.grad, dx_ref) < _DX_TOL[dtype]
        _assert_dw_close(weight.grad, dw_ref, dtype, name="dw")
        if bias is not None:
            _assert_dw_close(bias.grad, db_ref, dtype, name="db")
    else:
        x_ref = x.detach().requires_grad_()
        weight_ref = weight.detach().requires_grad_()
        bias_ref = bias.detach().requires_grad_() if bias is not None else None
        initial_ref = initial_states.detach().requires_grad_()
        ref = causal_conv1d_ref(
            torch.cat([initial_ref, x_ref], dim=-1),
            weight_ref,
            bias=bias_ref,
            activation="silu",
        )[..., W - 1 :]
        ref.backward(dout)
        assert _max_diff(x.grad, x_ref.grad) < _DX_TOL[dtype]
        assert _max_diff(initial_states.grad, initial_ref.grad) < _DX_TOL[dtype]
        _assert_dw_close(weight.grad, weight_ref.grad, dtype, name="dw")
        if bias is not None:
            _assert_dw_close(bias.grad, bias_ref.grad, dtype, name="db")


@pytest.mark.parametrize("offset", [1, 2, 3, 4, 8, 9])
@pytest.mark.parametrize("seqlen", [8, 9, 16, 1024])
def test_seq_sliced_view_fwd_bwd(device, dtype, offset, seqlen):
    """Gradients from an offset sequence view flow into the base slice."""
    B, D, W = 2, 8, 4
    storage_len = offset + seqlen + 3
    big = torch.randn(B, D, storage_len, dtype=dtype, device=device, requires_grad=True)
    x = big[..., offset : offset + seqlen]
    assert x.stride(-1) == 1
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(D, dtype=dtype, device=device, requires_grad=True)
    dout = torch.randn_like(x)

    out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias=bias, activation="silu")
    out_ref = causal_conv1d_ref(x, weight, bias=bias, activation="silu")
    assert _max_diff(out, out_ref) < _FWD_TOL[dtype]
    out.backward(dout)

    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, "silu")
    assert _max_diff(big.grad[..., offset : offset + seqlen], dx_ref) < _DX_TOL[dtype]
    assert torch.count_nonzero(big.grad[..., :offset]).item() == 0
    assert torch.count_nonzero(big.grad[..., offset + seqlen :]).item() == 0
    _assert_dw_close(weight.grad, dw_ref, dtype, name="dw")
    _assert_dw_close(bias.grad, db_ref, dtype, name="db")


@pytest.mark.parametrize("view_kind", ["channel", "batch"])
def test_outer_sliced_view_fwd_bwd(device, dtype, view_kind):
    """Channel/batch offsets and odd outer strides disable 16-byte rows."""
    if view_kind == "channel":
        base = torch.randn(2, 70, 37, dtype=dtype, device=device, requires_grad=True)
        x = base[:, 3:67, :]
    else:
        base = torch.randn(4, 63, 17, dtype=dtype, device=device, requires_grad=True)
        x = base[1:3]
    D = x.shape[1]
    weight = torch.randn(D, 4, dtype=dtype, device=device, requires_grad=True)
    dout = torch.randn_like(x)

    out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, activation="silu")
    out_ref = causal_conv1d_ref(x, weight, activation="silu")
    assert _max_diff(out, out_ref) < _FWD_TOL[dtype]
    out.backward(dout)

    dx_ref, dw_ref, _ = _ref_grads(x, weight, None, dout, "silu")
    actual_dx = base.grad[:, 3:67, :] if view_kind == "channel" else base.grad[1:3]
    assert _max_diff(actual_dx, dx_ref) < _DX_TOL[dtype]
    _assert_dw_close(weight.grad, dw_ref, dtype, name="dw")


def test_seq_sliced_dout_fwd_bwd(device, dtype):
    """Autograd passes a unit-inner-stride but misaligned dout view through."""
    B, D, L, W = 2, 16, 16, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(D, dtype=dtype, device=device, requires_grad=True)
    dout_big = torch.randn(B, D, L + 3, dtype=dtype, device=device)
    dout = dout_big[..., 1 : 1 + L]
    assert dout.stride(-1) == 1 and not dout.is_contiguous()

    out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias=bias, activation="silu")
    out.backward(dout)
    _assert_standard_grads(x, weight, bias, dout, "silu")


def test_update_offset_x_strided_state_and_weight(device, dtype):
    """Update honors scalar x/state/weight strides without vector promises."""
    B, D = 2, 16
    x_big = torch.randn(B, D, 9, dtype=dtype, device=device)
    x = x_big[..., 5:6]
    state_seed = torch.randn(B, D, 14, dtype=dtype, device=device)
    state_storage = state_seed.clone()
    state_ref_storage = state_seed.clone()
    state = state_storage[..., ::2]
    state_ref = state_ref_storage[..., ::2]
    weight_storage = torch.randn(D, 8, dtype=dtype, device=device)
    weight = weight_storage[:, 1::2]
    assert x.stride(-1) == 1
    assert state.stride(-1) == 2
    assert weight.stride(-1) == 2

    out = causal_conv1d_mojo.causal_conv1d_update(x, state, weight, activation="silu")
    out_ref = causal_conv1d_update_ref(x, state_ref, weight, activation="silu")

    assert _max_diff(out, out_ref) < _FWD_TOL[dtype]
    assert _max_diff(state, state_ref) < _FWD_TOL[dtype]
    assert _max_diff(state_storage, state_ref_storage) < _FWD_TOL[dtype]


def test_short_chunk_final_state_carries_initial_state_and_grad(device, dtype):
    """A sub-W chunk retains live history and routes its gradient back."""
    B, D, L, W = 1, 8, 1, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    initial = torch.randn(B, D, W - 1, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device)
    _, final = causal_conv1d_mojo.causal_conv1d_fn(
        x,
        weight,
        initial_states=initial,
        return_final_states=True,
        activation="silu",
    )
    expected = torch.cat([initial, x], dim=-1)[..., -(W - 1) :]
    assert _max_diff(final, expected) == 0.0

    dfinal = torch.randn_like(final)
    final.backward(dfinal)
    assert _max_diff(x.grad, dfinal[..., -L:]) == 0.0
    expected_dinitial = torch.zeros_like(initial)
    expected_dinitial[..., L:] = dfinal[..., : W - 1 - L]
    assert _max_diff(initial.grad, expected_dinitial) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only regression")
def test_cuda_one_token_offset_chunk_loop():
    """Thread final state through one-token views at every pointer offset."""
    B, D, L, W = 1, 64, 12, 4
    dtype = torch.float16
    x = torch.randn(B, D, L, dtype=dtype, device="cuda")
    weight = torch.randn(D, W, dtype=dtype, device="cuda")
    bias = torch.randn(D, dtype=dtype, device="cuda")
    state = torch.zeros(B, D, W - 1, dtype=dtype, device="cuda")

    chunks = []
    for t in range(L):
        out_t, state = causal_conv1d_mojo.causal_conv1d_fn(
            x[..., t : t + 1],
            weight,
            bias=bias,
            activation="silu",
            initial_states=state,
            return_final_states=True,
        )
        chunks.append(out_t)

    decoded = torch.cat(chunks, dim=-1)
    expected = causal_conv1d_ref(x, weight, bias=bias, activation="silu")
    assert _max_diff(decoded, expected) < _FWD_TOL[dtype]
