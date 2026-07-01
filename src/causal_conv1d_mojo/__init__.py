"""causal_conv1d_update, fused into a Mojo kernel and called via a direct
Python <-> Mojo CPython extension (no MAX framework). Trimmed repro of a
Mojo/Metal cold-cache bug — see repro.py."""

from __future__ import annotations

from causal_conv1d_mojo.update import causal_conv1d_update

__all__ = [
    "causal_conv1d_update",
]
