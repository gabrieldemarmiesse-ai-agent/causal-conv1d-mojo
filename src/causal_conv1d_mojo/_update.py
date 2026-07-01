"""`causal_conv1d_update` — single-step decode API (trimmed repro: mps
only, bias always present, silu activation always applied)."""

from __future__ import annotations

import torch

from causal_conv1d_mojo.update import native_update_mps


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    out = torch.empty_like(x)
    native_update_mps(x, weight, bias, conv_state, out)

    if unsqueeze:
        out = out.squeeze(-1)
    return out
