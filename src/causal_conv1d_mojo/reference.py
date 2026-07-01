"""Pure-pytorch reference for `causal_conv1d_update`."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_conv1d_update_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
    conv_state.copy_(x_new[:, :, -state_len:])
    x_unfolded = x_new.unfold(-1, width, 1)  # (batch, dim, L-width+1, width)
    out = (x_unfolded * weight.unsqueeze(0).unsqueeze(2)).sum(-1)[:, :, -seqlen:]
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)
