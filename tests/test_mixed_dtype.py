"""Mixed activation/parameter dtype coverage.

Upstream templates every kernel independently on ``input_t`` and
``weight_t``. These tests exercise all nine fp16/bf16/fp32 pairs on CPU
and CUDA, including the specialized channel-last forward and stateful
variants, then diff CUDA results against the installed upstream extension.
"""

from __future__ import annotations

from itertools import product

import pytest
import torch

from causal_conv1d_mojo import (
    causal_conv1d_fn,
    causal_conv1d_ref,
    causal_conv1d_update,
    causal_conv1d_update_ref,
)

from _helpers import (
    _DX_TOL,
    _FWD_TOL,
    _assert_dw_close,
    _expected_final_states,
    _max_diff,
    _ref_with_seq_idx,
)


_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}
_DTYPE_PAIRS = tuple(product(_DTYPES, repeat=2))


def _pair_id(pair):
    return f"x{_DTYPE_NAMES[pair[0]]}-w{_DTYPE_NAMES[pair[1]]}"


def _lower_precision_dtype(x_dtype, weight_dtype):
    """Tolerance key for arithmetic involving both input and weight."""
    if torch.bfloat16 in (x_dtype, weight_dtype):
        return torch.bfloat16
    if torch.float16 in (x_dtype, weight_dtype):
        return torch.float16
    return torch.float32


def _assert_fwd_close(actual, expected, tol_dtype, name):
    assert actual.dtype == expected.dtype
    diff = _max_diff(actual, expected)
    assert diff < _FWD_TOL[tol_dtype], (
        f"{name} max_diff={diff:.3e}, tol={_FWD_TOL[tol_dtype]:.3e}"
    )


def _upstream():
    # Import lazily: the upstream CUDA wheel is only part of the nvidia
    # extra, and CPU-only environments should still collect this file.
    import causal_conv1d

    return causal_conv1d


@pytest.mark.parametrize(
    ("x_dtype", "weight_dtype"), _DTYPE_PAIRS, ids=map(_pair_id, _DTYPE_PAIRS)
)
def test_mixed_fwd_contiguous(device, x_dtype, weight_dtype):
    B, D, L, W = 2, 16, 32, 4
    x = torch.randn(B, D, L, dtype=x_dtype, device=device)
    weight = torch.randn(D, W, dtype=weight_dtype, device=device)
    bias = torch.randn(D, dtype=weight_dtype, device=device)
    tol_dtype = _lower_precision_dtype(x_dtype, weight_dtype)

    out = causal_conv1d_fn(x, weight, bias=bias, activation="silu")
    ref = causal_conv1d_ref(x, weight, bias=bias, activation="silu")
    assert out.dtype == x.dtype
    _assert_fwd_close(out, ref, tol_dtype, "contiguous fwd")

    if device == "cuda" and x_dtype != weight_dtype:
        upstream = _upstream().causal_conv1d_fn(x, weight, bias=bias, activation="silu")
        _assert_fwd_close(out, upstream, tol_dtype, "upstream contiguous fwd")


@pytest.mark.parametrize(
    ("x_dtype", "weight_dtype"), _DTYPE_PAIRS, ids=map(_pair_id, _DTYPE_PAIRS)
)
def test_mixed_fwd_channel_last(device, x_dtype, weight_dtype):
    B, D, L, W = 2, 128, 19, 4
    x = torch.randn(B, L, D, dtype=x_dtype, device=device).transpose(1, 2)
    weight = torch.randn(D, W, dtype=weight_dtype, device=device)
    bias = torch.randn(D, dtype=weight_dtype, device=device)
    tol_dtype = _lower_precision_dtype(x_dtype, weight_dtype)
    assert x.stride(1) == 1 and x.stride(2) > 1

    out = causal_conv1d_fn(x, weight, bias=bias)
    ref = causal_conv1d_ref(x, weight, bias=bias)
    assert out.dtype == x.dtype
    _assert_fwd_close(out, ref, tol_dtype, "channel-last fwd")

    if device == "cuda" and x_dtype != weight_dtype:
        upstream = _upstream().causal_conv1d_fn(x, weight, bias=bias)
        _assert_fwd_close(out, upstream, tol_dtype, "upstream channel-last fwd")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only alignment gate")
def test_mixed_fwd_channel_last_bias_uses_weight_vector_alignment():
    """A 16B-but-not-32B fp32 bias must not enter the 32B vector path."""
    B, D, L, W = 1, 128, 16, 4
    x = torch.randn(B, L, D, dtype=torch.float16, device="cuda").transpose(1, 2)
    weight = torch.randn(D, W, dtype=torch.float32, device="cuda")
    bias_storage = torch.randn(D + 4, dtype=torch.float32, device="cuda")
    bias = bias_storage[4:]
    assert bias.is_contiguous()
    assert bias.data_ptr() % 16 == 0 and bias.data_ptr() % 32 != 0

    out = causal_conv1d_fn(x, weight, bias=bias)
    ref = causal_conv1d_ref(x, weight, bias=bias)
    _assert_fwd_close(out, ref, torch.float16, "misaligned mixed bias fallback")


@pytest.mark.parametrize(
    ("x_dtype", "weight_dtype"), _DTYPE_PAIRS, ids=map(_pair_id, _DTYPE_PAIRS)
)
def test_mixed_fwd_seq_idx(device, x_dtype, weight_dtype):
    B, D, L, W = 2, 16, 32, 4
    x = torch.randn(B, L, D, dtype=x_dtype, device=device).transpose(1, 2)
    weight = torch.randn(D, W, dtype=weight_dtype, device=device)
    bias = torch.randn(D, dtype=weight_dtype, device=device)
    seq_idx = torch.cat(
        [
            torch.zeros(B, 11, dtype=torch.int32, device=device),
            torch.ones(B, 13, dtype=torch.int32, device=device),
            torch.full((B, L - 24), 2, dtype=torch.int32, device=device),
        ],
        dim=1,
    )
    tol_dtype = _lower_precision_dtype(x_dtype, weight_dtype)

    out = causal_conv1d_fn(x, weight, bias=bias, seq_idx=seq_idx, activation="silu")
    ref = _ref_with_seq_idx(x, weight, bias, seq_idx, "silu")
    _assert_fwd_close(out, ref, tol_dtype, "seq_idx fwd")

    if device == "cuda" and x_dtype != weight_dtype:
        upstream = _upstream().causal_conv1d_fn(
            x, weight, bias=bias, seq_idx=seq_idx, activation="silu"
        )
        _assert_fwd_close(out, upstream, tol_dtype, "upstream seq_idx fwd")


@pytest.mark.parametrize(
    ("x_dtype", "weight_dtype"), _DTYPE_PAIRS, ids=map(_pair_id, _DTYPE_PAIRS)
)
def test_mixed_fwd_initial_and_final_states(device, x_dtype, weight_dtype):
    B, D, L, W = 2, 16, 13, 4
    x = torch.randn(B, L, D, dtype=x_dtype, device=device).transpose(1, 2)
    weight = torch.randn(D, W, dtype=weight_dtype, device=device)
    bias = torch.randn(D, dtype=weight_dtype, device=device)
    initial_states = torch.randn(B, W - 1, D, dtype=x_dtype, device=device).transpose(
        1, 2
    )
    tol_dtype = _lower_precision_dtype(x_dtype, weight_dtype)

    out, final_states = causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        initial_states=initial_states,
        return_final_states=True,
        activation="silu",
    )
    ref, ref_final = causal_conv1d_ref(
        x,
        weight,
        bias=bias,
        initial_states=initial_states,
        return_final_states=True,
        activation="silu",
    )
    assert out.dtype == x.dtype
    assert final_states.dtype == x.dtype
    _assert_fwd_close(out, ref, tol_dtype, "initial-states fwd")
    _assert_fwd_close(
        final_states, _expected_final_states(x, W), x_dtype, "final states"
    )
    _assert_fwd_close(final_states, ref_final, x_dtype, "reference final states")

    if device == "cuda" and x_dtype != weight_dtype:
        upstream_out, upstream_final = _upstream().causal_conv1d_fn(
            x,
            weight,
            bias=bias,
            initial_states=initial_states,
            return_final_states=True,
            activation="silu",
        )
        _assert_fwd_close(out, upstream_out, tol_dtype, "upstream initial-states fwd")
        _assert_fwd_close(
            final_states, upstream_final, x_dtype, "upstream final states"
        )


@pytest.mark.parametrize(
    ("x_dtype", "weight_dtype"), _DTYPE_PAIRS, ids=map(_pair_id, _DTYPE_PAIRS)
)
def test_mixed_backward(device, x_dtype, weight_dtype):
    B, D, L, W = 2, 16, 32, 4
    x = torch.randn(B, D, L, dtype=x_dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=weight_dtype, device=device, requires_grad=True)
    bias = torch.randn(D, dtype=weight_dtype, device=device, requires_grad=True)
    dout = torch.randn(B, D, L, dtype=x_dtype, device=device)
    tol_dtype = _lower_precision_dtype(x_dtype, weight_dtype)

    out = causal_conv1d_fn(x, weight, bias=bias, activation="silu")
    grads = torch.autograd.grad(out, (x, weight, bias), dout)

    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_()
    out_ref = causal_conv1d_ref(x_ref, weight_ref, bias=bias_ref, activation="silu")
    ref_grads = torch.autograd.grad(
        out_ref, (x_ref, weight_ref, bias_ref), dout.detach()
    )

    dx, dweight, dbias = grads
    dx_ref, dweight_ref, dbias_ref = ref_grads
    assert dx.dtype == x.dtype
    assert dweight.dtype == weight.dtype
    assert dbias.dtype == bias.dtype
    dx_diff = _max_diff(dx, dx_ref)
    assert dx_diff < _DX_TOL[tol_dtype], (
        f"dx max_diff={dx_diff:.3e}, tol={_DX_TOL[tol_dtype]:.3e}"
    )
    _assert_dw_close(dweight, dweight_ref, tol_dtype, name="mixed dweight")
    _assert_dw_close(dbias, dbias_ref, tol_dtype, name="mixed dbias")

    if device == "cuda" and x_dtype != weight_dtype:
        x_up = x.detach().clone().requires_grad_()
        weight_up = weight.detach().clone().requires_grad_()
        bias_up = bias.detach().clone().requires_grad_()
        out_up = _upstream().causal_conv1d_fn(
            x_up, weight_up, bias=bias_up, activation="silu"
        )
        up_grads = torch.autograd.grad(
            out_up, (x_up, weight_up, bias_up), dout.detach()
        )
        up_dx_diff = _max_diff(dx, up_grads[0])
        assert up_dx_diff < _DX_TOL[tol_dtype]
        _assert_dw_close(dweight, up_grads[1], tol_dtype, name="upstream dweight")
        _assert_dw_close(dbias, up_grads[2], tol_dtype, name="upstream dbias")


@pytest.mark.parametrize(
    ("x_dtype", "weight_dtype"), _DTYPE_PAIRS, ids=map(_pair_id, _DTYPE_PAIRS)
)
def test_mixed_update(device, x_dtype, weight_dtype):
    B, D, W, state_len = 3, 16, 4, 6
    weight = torch.randn(D, W, dtype=weight_dtype, device=device)
    bias = torch.randn(D, dtype=weight_dtype, device=device)
    tol_dtype = _lower_precision_dtype(x_dtype, weight_dtype)

    # Common decode shape: 2-D x and a linear state buffer.
    x_2d = torch.randn(B, D, dtype=x_dtype, device=device)
    state = torch.randn(B, D, state_len, dtype=x_dtype, device=device)
    state_ours = state.clone()
    state_ref = state.clone()
    out = causal_conv1d_update(x_2d, state_ours, weight, bias=bias, activation="silu")
    ref = causal_conv1d_update_ref(
        x_2d, state_ref, weight, bias=bias, activation="silu"
    )
    assert out.dtype == x_2d.dtype
    assert state_ours.dtype == x_2d.dtype
    _assert_fwd_close(out, ref, tol_dtype, "2-D update")
    _assert_fwd_close(state_ours, state_ref, x_dtype, "2-D update state")

    if device == "cuda" and x_dtype != weight_dtype:
        state_up = state.clone()
        out_up = _upstream().causal_conv1d_update(
            x_2d, state_up, weight, bias=bias, activation="silu"
        )
        _assert_fwd_close(out, out_up, tol_dtype, "upstream 2-D update")
        _assert_fwd_close(state_ours, state_up, x_dtype, "upstream 2-D state")

    # Short burst with both circular addressing and paged state rows.
    pool_size, seqlen = 5, 2
    x_3d = torch.randn(B, D, seqlen, dtype=x_dtype, device=device)
    pool = torch.randn(pool_size, D, state_len, dtype=x_dtype, device=device)
    indices = torch.tensor([3, 0, 4], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([1, 5, 7], dtype=torch.int32, device=device)
    pool_ours = pool.clone()
    out = causal_conv1d_update(
        x_3d,
        pool_ours,
        weight,
        bias=bias,
        cache_seqlens=cache_seqlens,
        conv_state_indices=indices,
    )

    selected_ref = pool.index_select(0, indices.to(torch.int64)).clone()
    ref = causal_conv1d_update_ref(
        x_3d,
        selected_ref,
        weight,
        bias=bias,
        cache_seqlens=cache_seqlens,
    )
    pool_ref = pool.clone()
    pool_ref.index_copy_(0, indices.to(torch.int64), selected_ref)
    assert out.dtype == x_3d.dtype
    assert pool_ours.dtype == x_3d.dtype
    _assert_fwd_close(out, ref, tol_dtype, "3-D circular indexed update")
    _assert_fwd_close(pool_ours, pool_ref, x_dtype, "paged circular state")

    if device == "cuda" and x_dtype != weight_dtype:
        pool_up = pool.clone()
        out_up = _upstream().causal_conv1d_update(
            x_3d,
            pool_up,
            weight,
            bias=bias,
            cache_seqlens=cache_seqlens,
            conv_state_indices=indices,
        )
        _assert_fwd_close(out, out_up, tol_dtype, "upstream 3-D update")
        _assert_fwd_close(pool_ours, pool_up, x_dtype, "upstream paged state")


@pytest.mark.parametrize("activation", [None, "silu"], ids=["no_silu", "silu"])
def test_mixed_width_bias_and_silu_sweep(device, width, bias_present, activation):
    """Exercise every width and both boolean kernel gates with AMP dtypes."""
    B, D, L = 1, 8, 16
    x_dtype, weight_dtype = torch.float16, torch.float32
    x = torch.randn(B, D, L, dtype=x_dtype, device=device, requires_grad=True)
    weight = torch.randn(
        D, width, dtype=weight_dtype, device=device, requires_grad=True
    )
    bias = (
        torch.randn(D, dtype=weight_dtype, device=device, requires_grad=True)
        if bias_present
        else None
    )
    inputs = (x, weight) if bias is None else (x, weight, bias)
    dout = torch.randn_like(x)

    out = causal_conv1d_fn(x, weight, bias=bias, activation=activation)
    grads = torch.autograd.grad(out, inputs, dout)
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    ref_inputs = (
        (x_ref, weight_ref) if bias_ref is None else (x_ref, weight_ref, bias_ref)
    )
    ref = causal_conv1d_ref(x_ref, weight_ref, bias=bias_ref, activation=activation)
    ref_grads = torch.autograd.grad(ref, ref_inputs, dout.detach())
    _assert_fwd_close(out, ref, torch.float16, "width-sweep fwd")
    assert grads[0].dtype == x.dtype
    assert grads[1].dtype == weight.dtype
    assert _max_diff(grads[0], ref_grads[0]) < _DX_TOL[torch.float16]
    _assert_dw_close(grads[1], ref_grads[1], torch.float16, name="width dweight")
    if bias is not None:
        assert grads[2].dtype == bias.dtype
        _assert_dw_close(grads[2], ref_grads[2], torch.float16, name="width dbias")

    state = torch.randn(B, D, width - 1, dtype=x_dtype, device=device)
    state_ours = state.clone()
    state_ref = state.clone()
    update_out = causal_conv1d_update(
        x.detach()[..., 0],
        state_ours,
        weight.detach(),
        bias=bias.detach() if bias is not None else None,
        activation=activation,
    )
    update_ref = causal_conv1d_update_ref(
        x.detach()[..., 0],
        state_ref,
        weight.detach(),
        bias=bias.detach() if bias is not None else None,
        activation=activation,
    )
    _assert_fwd_close(update_out, update_ref, torch.float16, "width-sweep update")
    _assert_fwd_close(state_ours, state_ref, torch.float16, "width-sweep state")


@pytest.mark.parametrize("op", ["fwd", "update"])
def test_bias_dtype_must_match_weight(op):
    x = torch.randn(1, 8, 16, dtype=torch.float16)
    weight = torch.randn(8, 4, dtype=torch.float32)
    bias = torch.randn(8, dtype=torch.bfloat16)
    match = (
        r"bias\.dtype \(torch\.bfloat16\) must match weight\.dtype \(torch\.float32\)"
    )
    with pytest.raises(NotImplementedError, match=match):
        if op == "fwd":
            causal_conv1d_fn(x, weight, bias=bias)
        else:
            state = torch.randn(1, 8, 3, dtype=x.dtype)
            causal_conv1d_update(x[..., 0], state, weight, bias=bias)


def test_unsupported_weight_dtype_raises():
    x = torch.randn(1, 8, 16, dtype=torch.float16)
    weight = torch.randn(8, 4, dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="unsupported weight dtype"):
        causal_conv1d_fn(x, weight)


def test_state_dtypes_follow_input():
    x = torch.randn(1, 8, 16, dtype=torch.float16)
    weight = torch.randn(8, 4, dtype=torch.float32)
    wrong_initial = torch.randn(1, 8, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="initial_states.dtype"):
        causal_conv1d_fn(x, weight, initial_states=wrong_initial)

    wrong_final = torch.empty(1, 8, 3, dtype=torch.float32)
    with pytest.raises(ValueError, match="final_states_out.dtype"):
        causal_conv1d_fn(
            x, weight, return_final_states=True, final_states_out=wrong_final
        )

    wrong_state = torch.randn(1, 8, 3, dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="conv_state.dtype"):
        causal_conv1d_update(x[..., 0], wrong_state, weight)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS fast path requires an Apple GPU"
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
def test_mps_small_shape_fallback_skipped_for_mixed_dtypes(x_dtype):
    """A mixed-dtype call must not change answer across the size threshold.

    Below `_MPS_FWD_FALLBACK_THRESHOLD` the wrapper normally hands off to
    `causal_conv1d_ref`, which mirrors upstream in rounding x through
    `weight.dtype` before convolving; the kernels widen x and the
    parameters to fp32 independently. Those agree only when the dtypes
    match, so the fast path is gated on `weight.dtype == x.dtype`. With
    weights *narrower* than x the fallback would lose ~1e-3 relative,
    which the fp32 tolerance below catches.
    """
    B, D, L, W = 1, 64, 128, 4
    assert B * D * L < 4 * 1024 * 1024, "shape must be inside the fallback window"
    weight_dtype = torch.float16

    x = torch.randn(B, D, L, dtype=x_dtype)
    weight = torch.randn(D, W, dtype=weight_dtype)
    bias = torch.randn(D, dtype=weight_dtype)

    got = causal_conv1d_fn(
        x.to("mps"), weight.to("mps"), bias=bias.to("mps"), activation="silu"
    ).cpu()
    expected = causal_conv1d_fn(x, weight, bias=bias, activation="silu")

    diff = _max_diff(got, expected)
    assert diff < _FWD_TOL[x_dtype], (
        f"mps mixed-dtype fwd took the rounding fallback: max_diff={diff:.3e}"
    )

    # Same gate on the decode path.
    conv_state = torch.randn(B, D, W - 1, dtype=x_dtype)
    tok = torch.randn(B, D, dtype=x_dtype)
    got_u = causal_conv1d_update(
        tok.to("mps"),
        conv_state.clone().to("mps"),
        weight.to("mps"),
        bias=bias.to("mps"),
        activation="silu",
    ).cpu()
    expected_u = causal_conv1d_update(
        tok, conv_state.clone(), weight, bias=bias, activation="silu"
    )
    diff_u = _max_diff(got_u, expected_u)
    assert diff_u < _FWD_TOL[x_dtype], (
        f"mps mixed-dtype update took the rounding fallback: max_diff={diff_u:.3e}"
    )
