"""causal_conv1d_update, fused into a Mojo kernel and called via a direct
Python <-> Mojo CPython extension (no MAX framework). Trimmed repro of a
Mojo/Metal cold-cache bug — see tests/test_update.py."""

from __future__ import annotations

from causal_conv1d_mojo._update import causal_conv1d_update
from causal_conv1d_mojo.reference import causal_conv1d_update_ref

__all__ = [
    "causal_conv1d_update",
    "causal_conv1d_update_ref",
]
