"""Deterministic backward coverage for CPU and GPU variants.

Upstream selects deterministic dweight/dbias reduction per backward call:
the ``CAUSAL_CONV1D_DETERMINISTIC`` environment override wins, otherwise
PyTorch's deterministic-algorithms setting applies. These tests exercise
both selectors, repeatability of every returned gradient, the feature
variants that write private workspace rows, and the atomic-path override.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import causal_conv1d_mojo
from causal_conv1d_mojo._fn import _use_deterministic_mode

from _helpers import (
    _DX_TOL,
    _assert_dw_close,
    _max_diff,
    _ref_grads,
    _ref_grads_with_seq_idx,
)


@contextmanager
def _deterministic_mode(mode, monkeypatch):
    """Select one upstream-compatible trigger and restore torch state."""
    was_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        if mode == "env":
            torch.use_deterministic_algorithms(False)
            monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
        else:
            monkeypatch.delenv("CAUSAL_CONV1D_DETERMINISTIC", raising=False)
            torch.use_deterministic_algorithms(True)
        assert _use_deterministic_mode()
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled)


def _clone_leaf(tensor):
    return tensor.detach().clone(memory_format=torch.preserve_format).requires_grad_()


def _run_backward(
    x_base,
    weight_base,
    bias_base,
    dout,
    *,
    activation=None,
    seq_idx=None,
    initial_states_base=None,
    dfinal_states=None,
):
    """Run one fresh graph and clone every differentiable input gradient."""
    x = _clone_leaf(x_base)
    weight = _clone_leaf(weight_base)
    bias = _clone_leaf(bias_base) if bias_base is not None else None
    initial_states = (
        _clone_leaf(initial_states_base) if initial_states_base is not None else None
    )

    result = causal_conv1d_mojo.causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        seq_idx=seq_idx,
        initial_states=initial_states,
        return_final_states=dfinal_states is not None,
        activation=activation,
    )
    if dfinal_states is None:
        result.backward(dout)
    else:
        out, final_states = result
        torch.autograd.backward((out, final_states), (dout, dfinal_states))

    assert x.grad is not None
    assert weight.grad is not None
    grads = {"dx": x.grad.detach().clone(), "dw": weight.grad.detach().clone()}
    if bias is not None:
        assert bias.grad is not None
        grads["db"] = bias.grad.detach().clone()
    if initial_states is not None:
        assert initial_states.grad is not None
        grads["dinitial_states"] = initial_states.grad.detach().clone()
    return grads


def _assert_repeated_equal(results):
    first = results[0]
    for run_idx, current in enumerate(results[1:], start=1):
        assert current.keys() == first.keys()
        for name in first:
            assert torch.equal(current[name], first[name]), (
                f"{name} changed between deterministic runs 0 and {run_idx}"
            )


@pytest.mark.parametrize("mode", ["env", "torch"], ids=["env1", "torch_flag"])
@pytest.mark.parametrize(
    "shape", [(64, 64, 1024), (7, 16, 73)], ids=["many_rows", "odd_seqlen"]
)
def test_deterministic_backward_repeatable_and_correct(
    device, dtype, mode, shape, monkeypatch
):
    """Both selectors are bit-repeatable on aligned and odd lengths."""
    B, D, L = shape
    W = 4
    if L % 2:
        # Channel-last keeps every row base 16-byte aligned even though L
        # is odd, while bwd still selects its unaligned, fully-strided leaf.
        x = torch.randn(B, L, D, dtype=dtype, device=device).transpose(1, 2)
        dout = torch.randn(B, L, D, dtype=dtype, device=device).transpose(1, 2)
    else:
        x = torch.randn(B, D, L, dtype=dtype, device=device)
        dout = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, W, dtype=dtype, device=device)
    bias = torch.randn(D, dtype=dtype, device=device)

    with _deterministic_mode(mode, monkeypatch):
        results = [
            _run_backward(x, weight, bias, dout, activation="silu") for _ in range(3)
        ]
        dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, "silu")

    _assert_repeated_equal(results)
    actual = results[0]
    assert _max_diff(actual["dx"], dx_ref) < _DX_TOL[dtype]
    _assert_dw_close(actual["dw"], dw_ref, dtype, name="deterministic dw")
    _assert_dw_close(actual["db"], db_ref, dtype, name="deterministic db")


def test_deterministic_no_bias_silu_seq_idx(device, monkeypatch):
    """No-bias + silu + seq_idx writes every private dweight row."""
    B, D, L, W = 8, 16, 67, 4
    dtype = torch.float16
    x = torch.randn(B, L, D, dtype=dtype, device=device).transpose(1, 2)
    weight = torch.randn(D, W, dtype=dtype, device=device)
    dout = torch.randn(B, L, D, dtype=dtype, device=device).transpose(1, 2)
    seq_idx = torch.cat(
        (
            torch.zeros(B, 19, dtype=torch.int32, device=device),
            torch.ones(B, 23, dtype=torch.int32, device=device),
            torch.full((B, L - 42), 2, dtype=torch.int32, device=device),
        ),
        dim=1,
    )

    with _deterministic_mode("env", monkeypatch):
        results = [
            _run_backward(
                x,
                weight,
                None,
                dout,
                activation="silu",
                seq_idx=seq_idx,
            )
            for _ in range(2)
        ]
        dx_ref, dw_ref, _ = _ref_grads_with_seq_idx(
            x, weight, None, seq_idx, dout, "silu"
        )

    _assert_repeated_equal(results)
    assert _max_diff(results[0]["dx"], dx_ref) < _DX_TOL[dtype]
    _assert_dw_close(results[0]["dw"], dw_ref, dtype, name="seq_idx dw")


def test_deterministic_initial_final_states_channel_last(device, monkeypatch):
    """Channel-last dx/dw/db/dinitial_states stay bit-identical.

    Supplying ``dfinal_states`` also exercises the deterministic conv dx
    followed by the direct final-state tail gradient update.
    """
    B, D, L, W = 8, 16, 65, 4
    dtype = torch.bfloat16
    x = torch.randn(B, L, D, dtype=dtype, device=device).transpose(1, 2)
    weight = torch.randn(D, W, dtype=dtype, device=device)
    bias = torch.randn(D, dtype=dtype, device=device)
    initial_states = torch.randn(B, D, W - 1, dtype=dtype, device=device)
    dout = torch.randn(B, L, D, dtype=dtype, device=device).transpose(1, 2)
    dfinal_states = torch.randn(B, D, W - 1, dtype=dtype, device=device)

    with _deterministic_mode("env", monkeypatch):
        results = [
            _run_backward(
                x,
                weight,
                bias,
                dout,
                activation="silu",
                initial_states_base=initial_states,
                dfinal_states=dfinal_states,
            )
            for _ in range(2)
        ]

        x_ref = _clone_leaf(x)
        weight_ref = _clone_leaf(weight)
        bias_ref = _clone_leaf(bias)
        initial_states_ref = _clone_leaf(initial_states)
        out_ref = causal_conv1d_mojo.causal_conv1d_ref(
            torch.cat((initial_states_ref, x_ref), dim=-1),
            weight_ref,
            bias=bias_ref,
            activation="silu",
        )[..., W - 1 :]
        final_states_ref = x_ref[..., -(W - 1) :]
        torch.autograd.backward((out_ref, final_states_ref), (dout, dfinal_states))

    _assert_repeated_equal(results)
    actual = results[0]
    assert _max_diff(actual["dx"], x_ref.grad) < _DX_TOL[dtype]
    _assert_dw_close(actual["dw"], weight_ref.grad, dtype, name="initial dw")
    _assert_dw_close(actual["db"], bias_ref.grad, dtype, name="initial db")
    assert (
        _max_diff(actual["dinitial_states"], initial_states_ref.grad) < _DX_TOL[dtype]
    )


@pytest.mark.parametrize(
    ("dtype", "width", "seqlen"),
    [
        pytest.param(torch.float16, 9, 1024, id="fp16_w9_aligned"),
        pytest.param(torch.float32, 5, 128, id="fp32_w5"),
    ],
)
def test_deterministic_wide_widths(device, dtype, width, seqlen, monkeypatch):
    """Wide-width deterministic leaves compile, repeat, and match ref."""
    B, D = 4, 16
    x = torch.randn(B, D, seqlen, dtype=dtype, device=device)
    weight = torch.randn(D, width, dtype=dtype, device=device)
    bias = torch.randn(D, dtype=dtype, device=device)
    dout = torch.randn_like(x)

    with _deterministic_mode("env", monkeypatch):
        results = [_run_backward(x, weight, bias, dout) for _ in range(2)]
        dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, None)

    _assert_repeated_equal(results)
    assert _max_diff(results[0]["dx"], dx_ref) < _DX_TOL[dtype]
    _assert_dw_close(results[0]["dw"], dw_ref, dtype, name="wide dw")
    _assert_dw_close(results[0]["db"], db_ref, dtype, name="wide db")


@pytest.mark.parametrize("shape", [(0, 8, 17), (4, 8, 0)])
def test_deterministic_zero_sized_edges(device, shape, monkeypatch):
    """Zero-batch/seqlen early returns reduce zero-filled workspaces."""
    B, D, L = shape
    dtype = torch.float32
    x = torch.randn(B, D, L, dtype=dtype, device=device)
    weight = torch.randn(D, 4, dtype=dtype, device=device)
    bias = torch.randn(D, dtype=dtype, device=device)
    dout = torch.randn_like(x)

    with _deterministic_mode("env", monkeypatch):
        results = [_run_backward(x, weight, bias, dout) for _ in range(2)]

    _assert_repeated_equal(results)
    assert torch.equal(results[0]["dw"], torch.zeros_like(weight))
    assert torch.equal(results[0]["db"], torch.zeros_like(bias))


def test_env_zero_overrides_torch_deterministic(device, monkeypatch):
    """Explicit ``=0`` selects the correct atomic variant under torch det."""
    was_enabled = torch.are_deterministic_algorithms_enabled()
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "0")
    torch.use_deterministic_algorithms(True)
    try:
        assert not _use_deterministic_mode()
        B, D, L, W = 8, 16, 64, 4
        dtype = torch.float32
        x = torch.randn(B, D, L, dtype=dtype, device=device)
        weight = torch.randn(D, W, dtype=dtype, device=device)
        bias = torch.randn(D, dtype=dtype, device=device)
        dout = torch.randn_like(x)
        actual = _run_backward(x, weight, bias, dout, activation="silu")
        dx_ref, dw_ref, db_ref = _ref_grads(x, weight, bias, dout, "silu")
    finally:
        torch.use_deterministic_algorithms(was_enabled)

    assert _max_diff(actual["dx"], dx_ref) < _DX_TOL[dtype]
    _assert_dw_close(actual["dw"], dw_ref, dtype, name="override dw")
    _assert_dw_close(actual["db"], db_ref, dtype, name="override db")


_COMPILE_BACKEND = "aot_eager" if torch.version.hip is not None else "inductor"


def test_torch_compile_fullgraph_deterministic_cuda(monkeypatch):
    """A deterministic-mode forward graph remains fullgraph traceable."""
    if not torch.cuda.is_available():
        pytest.skip("needs cuda")
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    B, D, L = 2, 64, 128
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda")
    weight = torch.randn(D, 4, dtype=torch.float16, device="cuda")
    bias = torch.randn(D, dtype=torch.float16, device="cuda")

    @torch.compile(fullgraph=True, backend=_COMPILE_BACKEND)
    def f(x, weight, bias):
        return causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias, activation="silu")

    out_compiled = f(x, weight, bias)
    out_eager = causal_conv1d_mojo.causal_conv1d_fn(x, weight, bias, activation="silu")
    assert torch.equal(out_compiled, out_eager)


def test_torch_compile_autograd_deterministic_cuda(monkeypatch):
    """Compiled autograd traces workspace allocation, custom op, and sum."""
    if not torch.cuda.is_available():
        pytest.skip("needs cuda")
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    B, D, L = 2, 64, 128
    x = torch.randn(B, D, L, dtype=torch.float16, device="cuda", requires_grad=True)
    weight = torch.randn(D, 4, dtype=torch.float16, device="cuda", requires_grad=True)
    bias = torch.randn(D, dtype=torch.float16, device="cuda", requires_grad=True)

    @torch.compile(backend=_COMPILE_BACKEND)
    def f(x, weight, bias):
        return causal_conv1d_mojo.causal_conv1d_fn(
            x, weight, bias, activation="silu"
        ).sum()

    f(x, weight, bias).backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert weight.grad is not None and weight.grad.shape == weight.shape
    assert bias.grad is not None and bias.grad.shape == bias.shape
