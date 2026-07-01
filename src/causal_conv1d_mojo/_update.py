"""`causal_conv1d_update` — single-step decode API (Apple mps only, in
this trimmed repro)."""

from __future__ import annotations

import torch

from causal_conv1d_mojo.update import native_update_mps


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    out = torch.empty_like(x)
    apply_silu = activation in ("silu", "swish")

    native_update_mps(x, weight, bias, conv_state, out, apply_silu)

    if unsqueeze:
        out = out.squeeze(-1)
    return out
