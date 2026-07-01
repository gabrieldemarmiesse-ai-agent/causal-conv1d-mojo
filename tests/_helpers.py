"""Test helpers for `test_update.py`."""

from __future__ import annotations

import torch

# Per-dtype tolerances.
_FWD_TOL = {torch.float16: 2e-2, torch.bfloat16: 2e-1, torch.float32: 1e-4}


def _make_bias(D, *, dtype, device, present, requires_grad=False):
    if not present:
        return None
    return torch.randn(D, dtype=dtype, device=device, requires_grad=requires_grad)


def _max_diff(a, b):
    assert torch.isfinite(a).all(), "actual contains NaN or Inf"
    assert torch.isfinite(b).all(), "reference contains NaN or Inf"
    return (a.float() - b.float()).abs().max().item()
