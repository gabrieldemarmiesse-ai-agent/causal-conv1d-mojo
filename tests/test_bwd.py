"""Backward-path tests for the pure-Mojo native extension.

Covers `causal_conv1d_mojo.causal_conv1d_fn` followed by `.backward()`:
the standard pytorch-reference dx/dw/db match, shape/dtype invariants,
the final_states gradient tail, initial_states gradients, seq_idx
segmented backward, the width sweep, and zero-sized tensors. The
forward-only tests live in `test_fwd.py`; the single-step update tests
live in `test_update.py`.

Each test runs on every available device + every supported dtype. CPU is
always present; CUDA is exercised only if a GPU is detected.
"""

import pytest
import torch

import causal_conv1d_mojo

from _helpers import (
    _DX_TOL,
    _assert_dw_close,
    _make_bias,
    _max_diff,
    _ref_grads,
    _ref_grads_with_seq_idx,
)


# ===---------- width sweep (2 / 3 / 4) ----------=== #


def test_width_backward(device, dtype, width, bias_present):
    B, D, L = 2, 32, 128
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, width, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn(B, D, L, dtype=dtype, device=device)

    out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias=bias, activation="silu")
    out.backward(dout)

    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, "silu")
    assert _max_diff(x.grad, dx_ref) < _DX_TOL[dtype]
    _assert_dw_close(weight.grad, dw_ref, dtype, name="dw")
    if bias_present:
        _assert_dw_close(bias.grad, db_ref, dtype, name="db")


# ===---------- backward / autograd ----------=== #


@pytest.mark.parametrize("shape", [(1, 8, 16), (2, 64, 128), (4, 256, 512)])
def test_backward_matches_pytorch_ref(device, dtype, shape, activation, bias_present):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn(B, D, L, dtype=dtype, device=device)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )
    out.backward(dout)

    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, activation)

    assert _max_diff(x.grad, dx_ref) < _DX_TOL[dtype], (
        f"dx max_diff={_max_diff(x.grad, dx_ref)}"
    )
    _assert_dw_close(weight.grad, dw_ref, dtype, name=f"dw (B*L={B * L})")
    if bias_present:
        _assert_dw_close(bias.grad, db_ref, dtype, name=f"db (B*L={B * L})")
    else:
        assert bias is None and db_ref is None


def test_backward_shapes_and_dtypes(device, dtype, activation, bias_present):
    B, D, L, W = 2, 64, 128, 4
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn_like(x)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )
    out.backward(dout)

    assert x.grad.shape == x.shape and x.grad.dtype == x.dtype
    assert weight.grad.shape == weight.shape and weight.grad.dtype == weight.dtype
    if bias_present:
        assert bias.grad.shape == bias.shape and bias.grad.dtype == bias.dtype


# ===---------- dedicated channel-last backward ----------=== #


_CHANNEL_LAST_CASES = [
    pytest.param(torch.float16, 2, 131, True, "silu", id="fp16-w2-tail"),
    pytest.param(torch.bfloat16, 3, 131, False, None, id="bf16-w3-tail"),
    pytest.param(torch.float32, 4, 2, False, None, id="fp32-w4-short"),
    pytest.param(torch.float16, 5, 67, True, "silu", id="fp16-w5-tail"),
]


@pytest.mark.parametrize(
    "case_dtype,width,seqlen,bias_present,activation", _CHANNEL_LAST_CASES
)
def test_channel_last_backward_matches_references(
    device, case_dtype, width, seqlen, bias_present, activation
):
    """The channel-vectorized kernel matches PyTorch and upstream CUDA.

    Cases cover every supported fast-path width, all three dtypes, batch
    greater than one, L tails, and L < W-1. ``dout`` is deliberately
    seqlen-contiguous: the autograd wrapper must convert it to the same
    channel-last layout family as x before dispatch.
    """
    B, D = 2, 32
    x = (
        torch.randn(B, seqlen, D, dtype=case_dtype, device=device)
        .transpose(1, 2)
        .requires_grad_()
    )
    assert x.stride(1) == 1 and x.stride(2) > 1
    weight = torch.randn(D, width, dtype=case_dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D,
        dtype=case_dtype,
        device=device,
        present=bias_present,
        requires_grad=True,
    )
    dout = torch.randn(B, D, seqlen, dtype=case_dtype, device=device)
    assert dout.stride(-1) == 1

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )
    out.backward(dout)

    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, activation)
    assert _max_diff(x.grad, dx_ref) < _DX_TOL[case_dtype]
    _assert_dw_close(weight.grad, dw_ref, case_dtype, name="dw[channel-last]")
    if bias is not None:
        _assert_dw_close(bias.grad, db_ref, case_dtype, name="db[channel-last]")

    # The installed reference supports widths 2..4 and channel-last CUDA.
    if device == "cuda" and width <= 4:
        from causal_conv1d import causal_conv1d_fn as upstream_fn

        x_up = x.detach().clone(memory_format=torch.preserve_format).requires_grad_()
        w_up = weight.detach().clone().requires_grad_()
        b_up = bias.detach().clone().requires_grad_() if bias is not None else None
        assert x_up.stride(1) == 1
        out_up = upstream_fn(x_up, w_up, b_up, activation=activation)
        out_up.backward(dout)
        assert _max_diff(x.grad, x_up.grad) < _DX_TOL[case_dtype]
        _assert_dw_close(
            weight.grad, w_up.grad, case_dtype, name="dw[channel-last upstream]"
        )
        if bias is not None:
            _assert_dw_close(
                bias.grad, b_up.grad, case_dtype, name="db[channel-last upstream]"
            )


@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("seq_pattern", ["cross_chunk", "with_padding"])
def test_channel_last_seq_idx_backward(device, activation, seq_pattern):
    """Packed ids, including padding and a boundary at the 128-row seam."""
    B, D, L, W = 2, 32, 133, 4
    dtype = torch.float16
    x = (
        torch.randn(B, L, D, dtype=dtype, device=device)
        .transpose(1, 2)
        .requires_grad_()
    )
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(D, dtype=dtype, device=device, requires_grad=True)
    dout = torch.randn_like(x)
    if seq_pattern == "cross_chunk":
        seq_idx = torch.stack(
            [
                torch.cat(
                    [
                        torch.zeros(129, dtype=torch.int32, device=device),
                        torch.ones(4, dtype=torch.int32, device=device),
                    ]
                ),
                torch.cat(
                    [
                        torch.full((63,), 3, dtype=torch.int32, device=device),
                        torch.full((70,), 4, dtype=torch.int32, device=device),
                    ]
                ),
            ]
        )
    else:
        seq_idx = torch.stack(
            [
                torch.cat(
                    [
                        torch.zeros(31, dtype=torch.int32, device=device),
                        torch.full((5,), -1, dtype=torch.int32, device=device),
                        torch.ones(97, dtype=torch.int32, device=device),
                    ]
                ),
                torch.cat(
                    [
                        torch.full((2,), -1, dtype=torch.int32, device=device),
                        torch.full((64,), 7, dtype=torch.int32, device=device),
                        torch.full((67,), 8, dtype=torch.int32, device=device),
                    ]
                ),
            ]
        )

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, seq_idx=seq_idx, activation=activation
    )
    out.backward(dout)
    dx_ref, dw_ref, db_ref = _ref_grads_with_seq_idx(
        x, weight, bias, seq_idx, dout, activation
    )
    assert _max_diff(x.grad, dx_ref) < _DX_TOL[dtype]
    _assert_dw_close(weight.grad, dw_ref, dtype, name=f"dw[{seq_pattern}]")
    _assert_dw_close(bias.grad, db_ref, dtype, name=f"db[{seq_pattern}]")


# ===---------- final_states backward ----------=== #


def test_final_states_backward(device, dtype, width, bias_present):
    """Gradient w.r.t. final_states is added to the matching tail of dx.

    final_states[b, c, i] = x[b, c, seqlen - (W-1) + i] for the
    seqlen >= W-1 case, so dfinal_states[b, c, i] flows directly back
    into dx[b, c, seqlen - (W-1) + i] in addition to the conv-path dx
    contribution.
    """
    B, D, L, W = 2, 16, 64, width
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn(B, D, L, dtype=dtype, device=device)
    dfs = torch.randn(B, D, W - 1, dtype=dtype, device=device)

    out, fs = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation="silu", return_final_states=True
    )
    torch.autograd.backward([out, fs], [dout, dfs])

    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, "silu")
    dx_ref = dx_ref.clone()
    dx_ref[..., -(W - 1) :] += dfs.to(dx_ref.dtype)

    assert _max_diff(x.grad, dx_ref) < _DX_TOL[dtype], (
        f"dx max_diff={_max_diff(x.grad, dx_ref)}"
    )
    _assert_dw_close(weight.grad, dw_ref, dtype, name="dw")
    if bias_present:
        _assert_dw_close(bias.grad, db_ref, dtype, name="db")


@pytest.mark.parametrize("seqlen", [0, 1, 2])
def test_final_states_backward_short_seqlen_with_initial_states(device, dtype, seqlen):
    """seqlen < W-1 with initial_states: final_states copies
    `initial_states[..., seqlen:]` into its first W-1-seqlen slots, so
    dfinal_states must flow into dinitial_states there (upstream's
    `dxinit_vals[i] += dfinal_states[i - seqlen]`), and into dx's tail
    for the rest.
    """
    from causal_conv1d_mojo import causal_conv1d_ref

    B, D, W = 2, 16, 4
    x = torch.randn(B, D, seqlen, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    init = torch.randn(B, D, W - 1, dtype=dtype, device=device, requires_grad=True)
    dout = torch.randn(B, D, seqlen, dtype=dtype, device=device)
    dfs = torch.randn(B, D, W - 1, dtype=dtype, device=device)

    out, fs = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, initial_states=init, activation="silu", return_final_states=True
    )
    torch.autograd.backward([out, fs], [dout, dfs])

    # Reference: conv over cat([init, x]) + the state window, autograd.
    x_ref = x.detach().clone().requires_grad_()
    w_ref = weight.detach().clone().requires_grad_()
    init_ref = init.detach().clone().requires_grad_()
    full = torch.cat([init_ref, x_ref], dim=-1)
    out_ref = causal_conv1d_ref(full, w_ref, activation="silu")[..., W - 1 :]
    fs_ref = full[..., -(W - 1) :]
    torch.autograd.backward([out_ref, fs_ref], [dout, dfs])

    assert _max_diff(init.grad, init_ref.grad) < _DX_TOL[dtype], (
        f"dinit max_diff={_max_diff(init.grad, init_ref.grad)}"
    )
    if seqlen > 0:
        assert _max_diff(x.grad, x_ref.grad) < _DX_TOL[dtype], (
            f"dx max_diff={_max_diff(x.grad, x_ref.grad)}"
        )
    _assert_dw_close(weight.grad, w_ref.grad, dtype, name="dw")


def test_channel_last_short_final_states_backward(device):
    """dfinal_states is folded into channel-last dx when L < W-1."""
    B, D, L, W = 2, 32, 2, 5
    dtype = torch.float16
    x = (
        torch.randn(B, L, D, dtype=dtype, device=device)
        .transpose(1, 2)
        .requires_grad_()
    )
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(D, dtype=dtype, device=device, requires_grad=True)
    dout = torch.randn_like(x)
    dfinal = torch.randn(B, D, W - 1, dtype=dtype, device=device)

    out, final = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation="silu", return_final_states=True
    )
    torch.autograd.backward((out, final), (dout, dfinal))

    dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, "silu")
    dx_ref = dx_ref.clone()
    dx_ref[..., -L:] += dfinal[..., -L:]
    assert _max_diff(x.grad, dx_ref) < _DX_TOL[dtype]
    _assert_dw_close(weight.grad, dw_ref, dtype, name="dw[final channel-last]")
    _assert_dw_close(bias.grad, db_ref, dtype, name="db[final channel-last]")


# ===---------- initial_states backward ----------=== #


def test_initial_states_backward(device, dtype, width, bias_present):
    """Backward through initial_states: dx, dw, dbias, and dinitial_states
    all match the cat([initial_states, x]) reference. The kernel reads
    initial_states for the silu' recomputation in chunk 0 / tidx 0 and
    accumulates the boundary dweight contribution; dinitial_states is
    derived from dpre[0..W-2] with the anti-causal weight kernel.
    """
    from causal_conv1d_mojo import causal_conv1d_ref

    B, D, L = 2, 16, 64
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, width, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    init = torch.randn(B, D, width - 1, dtype=dtype, device=device, requires_grad=True)
    dout = torch.randn(B, D, L, dtype=dtype, device=device)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, initial_states=init, activation="silu"
    )
    out.backward(dout)

    # Reference: cat([init, x], -1) -> standard causal_conv1d_ref ->
    # autograd. Slice the resulting dx into dinit + dx parts.
    x_ref = x.detach().clone().requires_grad_()
    w_ref = weight.detach().clone().requires_grad_()
    b_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    init_ref = init.detach().clone().requires_grad_()
    out_ref = causal_conv1d_ref(
        torch.cat([init_ref, x_ref], dim=-1),
        w_ref,
        bias=b_ref,
        initial_states=None,
        activation="silu",
    )[..., width - 1 :]
    out_ref.backward(dout)

    assert _max_diff(x.grad, x_ref.grad) < _DX_TOL[dtype], (
        f"dx max_diff={_max_diff(x.grad, x_ref.grad)}"
    )
    _assert_dw_close(weight.grad, w_ref.grad, dtype, name="dw")
    if bias_present:
        _assert_dw_close(bias.grad, b_ref.grad, dtype, name="db")
    # dinitial_states correctness — main feature. Tighter tolerance than
    # dweight since the sum is over much fewer terms (W-1, not B*L).
    assert _max_diff(init.grad, init_ref.grad) < _DX_TOL[dtype], (
        f"dinit max_diff={_max_diff(init.grad, init_ref.grad)}"
    )


def test_channel_last_initial_states_backward(device):
    """The dedicated path computes dinit without packed-sequence gating."""
    from causal_conv1d_mojo import causal_conv1d_ref

    B, D, L, W = 2, 32, 67, 4
    dtype = torch.float16
    x = (
        torch.randn(B, L, D, dtype=dtype, device=device)
        .transpose(1, 2)
        .requires_grad_()
    )
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = torch.randn(D, dtype=dtype, device=device, requires_grad=True)
    init = torch.randn(B, D, W - 1, dtype=dtype, device=device, requires_grad=True)
    dout = torch.randn_like(x)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, initial_states=init, activation="silu"
    )
    out.backward(dout)

    x_ref = x.detach().clone(memory_format=torch.preserve_format).requires_grad_()
    w_ref = weight.detach().clone().requires_grad_()
    b_ref = bias.detach().clone().requires_grad_()
    init_ref = init.detach().clone().requires_grad_()
    out_ref = causal_conv1d_ref(
        torch.cat([init_ref, x_ref], dim=-1),
        w_ref,
        bias=b_ref,
        activation="silu",
    )[..., W - 1 :]
    out_ref.backward(dout)

    assert _max_diff(x.grad, x_ref.grad) < _DX_TOL[dtype]
    _assert_dw_close(weight.grad, w_ref.grad, dtype, name="dw[init channel-last]")
    _assert_dw_close(bias.grad, b_ref.grad, dtype, name="db[init channel-last]")
    assert _max_diff(init.grad, init_ref.grad) < _DX_TOL[dtype]


# ===---------- seq_idx backward ----------=== #


@pytest.mark.parametrize(
    "seq_idx_pattern", ["single", "two_equal", "varied", "with_padding"]
)
@pytest.mark.parametrize("channel_last", [False, True], ids=["contig", "channel_last"])
def test_seq_idx_backward(
    device, dtype, seq_idx_pattern, activation, bias_present, channel_last
):
    """Backward through seq_idx: dx/dw/db match the segmented reference.

    For each seq_idx run, only positions in that run contributed to
    each other in the forward; padding positions produced zero output
    so their dpre is zero. The backward must reproduce that segmented
    flow.
    """
    B, D, L, W = 2, 16, 64, 4
    if channel_last:
        x = (
            torch.randn(B, L, D, dtype=dtype, device=device)
            .transpose(1, 2)
            .requires_grad_()
        )
        assert x.stride(1) == 1 and x.stride(2) != 1
    else:
        x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn(B, D, L, dtype=dtype, device=device)

    if seq_idx_pattern == "single":
        seq_idx = torch.zeros(B, L, dtype=torch.int32, device=device)
    elif seq_idx_pattern == "two_equal":
        seq_idx = torch.cat(
            [
                torch.zeros(B, L // 2, dtype=torch.int32, device=device),
                torch.ones(B, L - L // 2, dtype=torch.int32, device=device),
            ],
            dim=1,
        )
    elif seq_idx_pattern == "varied":
        seq_idx = torch.cat(
            [
                torch.zeros(B, 10, dtype=torch.int32, device=device),
                torch.ones(B, 25, dtype=torch.int32, device=device),
                torch.full((B, L - 35), 2, dtype=torch.int32, device=device),
            ],
            dim=1,
        )
    else:  # with_padding
        seq_idx = torch.cat(
            [
                torch.zeros(B, 16, dtype=torch.int32, device=device),
                torch.full((B, 16), -1, dtype=torch.int32, device=device),
                torch.ones(B, L - 32, dtype=torch.int32, device=device),
            ],
            dim=1,
        )

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, seq_idx=seq_idx, activation=activation
    )
    out.backward(dout)

    dx_ref, dw_ref, db_ref = _ref_grads_with_seq_idx(
        x, weight, bias, seq_idx, dout, activation
    )

    assert _max_diff(x.grad, dx_ref) < _DX_TOL[dtype], (
        f"dx max_diff={_max_diff(x.grad, dx_ref)}, pattern={seq_idx_pattern}"
    )
    _assert_dw_close(weight.grad, dw_ref, dtype, name=f"dw[{seq_idx_pattern}]")
    if bias_present:
        _assert_dw_close(bias.grad, db_ref, dtype, name=f"db[{seq_idx_pattern}]")


# ===---------- zero-sized tensors ----------=== #


@pytest.mark.parametrize("shape", [(0, 64, 128), (2, 0, 128), (2, 64, 0), (0, 0, 0)])
def test_zero_sized_backward(device, dtype, shape, activation, bias_present):
    B, D, L = shape
    W = 4
    x = torch.randn(B, D, L, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, dtype=dtype, device=device, requires_grad=True)
    bias = _make_bias(
        D, dtype=dtype, device=device, present=bias_present, requires_grad=True
    )
    dout = torch.randn_like(x)

    out = causal_conv1d_mojo.causal_conv1d_fn(
        x, weight, bias=bias, activation=activation
    )
    out.backward(dout)

    assert x.grad.shape == x.shape and x.grad.dtype == x.dtype
    assert weight.grad.shape == weight.shape and weight.grad.dtype == weight.dtype
    if bias_present:
        assert bias.grad.shape == bias.shape and bias.grad.dtype == bias.dtype


@pytest.mark.parametrize("shape", [(0, 32, 7), (2, 32, 0)])
def test_zero_sized_channel_last_backward(device, shape):
    """Zero batch/L channel-last views return without launching grid zero."""
    B, D, L = shape
    dtype = torch.float16
    x = (
        torch.randn(B, L, D, dtype=dtype, device=device)
        .transpose(1, 2)
        .requires_grad_()
    )
    weight = torch.randn(D, 4, dtype=dtype, device=device, requires_grad=True)
    out = causal_conv1d_mojo.causal_conv1d_fn(x, weight, activation="silu")
    out.backward(torch.randn_like(out))
    assert x.grad is not None and x.grad.shape == x.shape
    assert weight.grad is not None and weight.grad.shape == weight.shape


def test_torch_compile_channel_last_autograd_cuda():
    """Dynamo/Inductor preserves the dedicated backward dispatch."""
    if not torch.cuda.is_available():
        pytest.skip("needs cuda")
    B, D, L = 2, 32, 133
    x = (
        torch.randn(B, L, D, dtype=torch.float16, device="cuda")
        .transpose(1, 2)
        .requires_grad_()
    )
    weight = torch.randn(D, 4, dtype=torch.float16, device="cuda", requires_grad=True)
    bias = torch.randn(D, dtype=torch.float16, device="cuda", requires_grad=True)
    backend = "aot_eager" if torch.version.hip is not None else "inductor"

    @torch.compile(backend=backend)
    def loss_fn(x, weight, bias):
        return causal_conv1d_mojo.causal_conv1d_fn(
            x, weight, bias, activation="silu"
        ).sum()

    loss_fn(x, weight, bias).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert bias.grad is not None and torch.isfinite(bias.grad).all()
