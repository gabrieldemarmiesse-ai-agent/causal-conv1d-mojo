"""Tests for `causal_conv1d_varlen_states` — the packed-batch state
extractor.

Upstream's optimized version uses a Triton kernel; ours uses a batched
PyTorch gather. Our independent `_ref` loop is the correctness oracle
for the randomized edge-case sweep.
"""

from __future__ import annotations

import random

import pytest
import torch

import causal_conv1d_mojo


def _make_packed(
    cu_seqlens: list[int],
    dim: int,
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    cu_dtype: torch.dtype = torch.int32,
    cu_device: str | torch.device | None = None,
    noncontiguous: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (total_tokens, dim) packed-batch tensor where each
    sequence's tokens contain easy-to-verify monotonically-increasing
    values."""
    total = cu_seqlens[-1]
    if noncontiguous:
        storage = torch.arange(total * dim * 2, dtype=dtype, device=device).reshape(
            total, dim * 2
        )
        x = storage[:, ::2]
    else:
        x = torch.arange(total * dim, dtype=dtype, device=device).reshape(total, dim)
    return x, torch.tensor(
        cu_seqlens,
        dtype=cu_dtype,
        device=device if cu_device is None else cu_device,
    )


def _cu_seqlens(lengths: list[int]) -> list[int]:
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return boundaries


def test_varlen_states_basic_cpu():
    """Three sequences of varying length, dim=2, state_len=3. Sequence
    lengths 5, 2, 4 → expect last 3 / all 2 (zero-padded) / last 3 of
    each respectively."""
    x, cu = _make_packed([0, 5, 7, 11], dim=2, dtype=torch.float32, device="cpu")
    out = causal_conv1d_mojo.causal_conv1d_varlen_states(x, cu, state_len=3)
    assert out.shape == (3, 2, 3)
    assert out.dtype == torch.float32

    # Seq 0: tokens 0..4, last 3 are tokens 2,3,4 → values rows 2,3,4 of x.
    expected_0 = x[2:5].T
    assert torch.equal(out[0], expected_0)

    # Seq 1: tokens 5,6 (only 2 tokens, state_len=3 → left-zero-pad).
    assert torch.equal(out[1, :, 0], torch.zeros(2))
    assert torch.equal(out[1, :, 1:], x[5:7].T)

    # Seq 2: tokens 7..10, last 3 are 8,9,10.
    expected_2 = x[8:11].T
    assert torch.equal(out[2], expected_2)


def test_varlen_states_zero_length_sequence():
    """Empty middle sequence — state should be entirely zero-padded."""
    x, cu = _make_packed([0, 3, 3, 5], dim=4, dtype=torch.float32, device="cpu")
    out = causal_conv1d_mojo.causal_conv1d_varlen_states(x, cu, state_len=2)
    assert out.shape == (3, 4, 2)
    assert torch.equal(out[1], torch.zeros(4, 2))


def test_varlen_states_state_longer_than_all_sequences():
    """state_len > every sequence — every output should be left-padded."""
    x, cu = _make_packed([0, 2, 4], dim=3, dtype=torch.float16, device="cpu")
    out = causal_conv1d_mojo.causal_conv1d_varlen_states(x, cu, state_len=8)
    assert out.shape == (2, 3, 8)
    # 6 zero columns on the left, then 2 columns from x.
    assert torch.equal(out[0, :, :6], torch.zeros(3, 6, dtype=torch.float16))
    assert torch.equal(out[0, :, 6:], x[0:2].T)
    assert torch.equal(out[1, :, :6], torch.zeros(3, 6, dtype=torch.float16))
    assert torch.equal(out[1, :, 6:], x[2:4].T)


def test_varlen_states_matches_ref(device):
    """The vectorized entry point and independent loop agree."""
    x, cu = _make_packed([0, 5, 5, 7, 15], dim=5, dtype=torch.float32, device=device)
    actual = causal_conv1d_mojo.causal_conv1d_varlen_states(x, cu, state_len=6)
    expected = causal_conv1d_mojo.causal_conv1d_varlen_states_ref(x, cu, state_len=6)

    assert (
        causal_conv1d_mojo.causal_conv1d_varlen_states
        is not causal_conv1d_mojo.causal_conv1d_varlen_states_ref
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_varlen_states_randomized(device, dtype):
    """Cross-check dtypes, index dtypes, layouts, and empty shapes."""
    rng = random.Random(0)
    cases = [
        ([], 5, 3),  # batch == 0
        ([0], 7, 4),  # total_tokens == 0
        ([0, 0, 0], 11, 5),  # several zero-length sequences
        ([1, 2, 0, 4], 12, 7),  # state longer than every sequence
        ([8, 0, 3, 17], 0, 6),  # empty state axis
    ]
    for _ in range(32):
        lengths = [rng.randrange(9) for _ in range(rng.randrange(13))]
        cases.append((lengths, rng.randrange(17), rng.choice([2, 3, 5, 9])))

    for case_idx, (lengths, state_len, dim) in enumerate(cases):
        cu_dtype = torch.int32 if case_idx % 2 == 0 else torch.int64
        # CUDA accepts CPU boundaries too; alternate them into the sweep.
        cu_device = "cpu" if device == "cuda" and case_idx % 3 == 0 else device
        noncontiguous = case_idx % 2 == 1
        x, cu = _make_packed(
            _cu_seqlens(lengths),
            dim,
            dtype=dtype,
            device=device,
            cu_dtype=cu_dtype,
            cu_device=cu_device,
            noncontiguous=noncontiguous,
        )

        actual = causal_conv1d_mojo.causal_conv1d_varlen_states(x, cu, state_len)
        expected = causal_conv1d_mojo.causal_conv1d_varlen_states_ref(x, cu, state_len)

        assert actual.shape == (len(lengths), dim, state_len)
        assert actual.dtype == dtype
        assert actual.device == x.device
        assert actual.stride(1) == 1
        assert torch.equal(actual, expected), (
            f"case={case_idx}, lengths={lengths}, state_len={state_len}, "
            f"dim={dim}, dtype={dtype}, device={device}, cu_dtype={cu_dtype}, "
            f"cu_device={cu_device}, noncontiguous={noncontiguous}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
@pytest.mark.filterwarnings(
    "ignore:Synchronization debug mode is a prototype feature.*:UserWarning"
)
def test_varlen_states_cuda_has_no_host_sync():
    """The vectorized CUDA path never reads a device scalar on the host."""
    x, cu = _make_packed(
        _cu_seqlens([0, 13, 2, 0, 31, 4]),
        dim=16,
        dtype=torch.float16,
        device="cuda",
        noncontiguous=True,
    )
    expected = causal_conv1d_mojo.causal_conv1d_varlen_states_ref(x, cu, state_len=19)

    try:
        torch.cuda.set_sync_debug_mode("error")
        actual = causal_conv1d_mojo.causal_conv1d_varlen_states(x, cu, state_len=19)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert torch.equal(actual, expected)
    assert actual.stride(1) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_varlen_states_matches_upstream_cuda():
    """Spot-check we match upstream's Triton kernel on CUDA — same
    semantics, same output dtype + layout."""
    pytest.importorskip("causal_conv1d.causal_conv1d_varlen")
    from causal_conv1d.causal_conv1d_varlen import (
        causal_conv1d_varlen_states as upstream_fn,
    )

    torch.manual_seed(0)
    cu = torch.tensor([0, 17, 17, 32, 80], dtype=torch.int32, device="cuda")
    x = torch.randn(80, 64, dtype=torch.float16, device="cuda")
    state_len = 10

    ours = causal_conv1d_mojo.causal_conv1d_varlen_states(x, cu, state_len)
    theirs = upstream_fn(x, cu, state_len)

    assert ours.shape == theirs.shape
    assert ours.dtype == theirs.dtype
    assert ours.stride(1) == 1
    assert torch.equal(ours, theirs)
