"""Variable-length packed-batch helpers — `causal_conv1d_varlen_states`.

Mirrors upstream's
`causal_conv1d.causal_conv1d_varlen.causal_conv1d_varlen_states`.

Given a packed batch of variable-length sequences laid out as a single
`(total_tokens, dim)` tensor with cumulative sequence lengths in
`cu_seqlens`, extract the last `state_len` tokens of each sequence
into a `(batch, dim, state_len)` tensor — the conv-state input format.
Shorter-than-`state_len` sequences are zero-padded on the left.

This is pure data movement (gather + zero fill); no conv and no Mojo
kernel. The primary implementation constructs every sequence's trailing
indices at once and issues a single batched PyTorch gather, avoiding the
device-to-host synchronization that scalar CUDA tensor indexing would
otherwise impose for every sequence. `causal_conv1d_varlen_states_ref`
keeps the simple per-sequence loop as an independent correctness oracle.
"""

from __future__ import annotations

import torch


def causal_conv1d_varlen_states(
    x: torch.Tensor, cu_seqlens: torch.Tensor, state_len: int
) -> torch.Tensor:
    """Extract the trailing `state_len` tokens of each packed sequence.

    Args:
        x: (total_tokens, dim) packed batch of token activations.
        cu_seqlens: (batch + 1,) cumulative sequence lengths, starting
            at 0. Must be sorted non-decreasing.
        state_len: number of trailing tokens per sequence to copy into
            the output state. Sequences shorter than `state_len` get
            left zero-padded.

    Returns:
        states: (batch, dim, state_len), `dtype` and `device` matching
        `x`. Dim is contiguous (`states.stride(1) == 1`), matching
        upstream's Triton output and the layout expected by
        `causal_conv1d_update`.
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.to(device=x.device).contiguous()

    # Shape checks use host-resident metadata, not tensor values. An empty x
    # has no row that can safely stand in for the masked gather below.
    if batch == 0 or x.shape[0] == 0:
        return torch.zeros(
            batch, state_len, dim, dtype=x.dtype, device=x.device
        ).transpose(1, 2)

    start = cu_seqlens[:-1]
    end = cu_seqlens[1:]
    offsets = torch.arange(state_len, dtype=cu_seqlens.dtype, device=x.device)
    idx = end[:, None] - state_len + offsets
    valid = idx >= start[:, None]

    # Invalid negative indices are clamped solely to make the gather safe;
    # masking also removes in-range rows that belong to the prior sequence.
    gathered = x[idx.clamp_min(0)]
    states = gathered.masked_fill(~valid[:, :, None], 0)
    return states.transpose(1, 2)


def causal_conv1d_varlen_states_ref(
    x: torch.Tensor, cu_seqlens: torch.Tensor, state_len: int
) -> torch.Tensor:
    """Reference trailing-state extraction using a per-sequence loop."""
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    boundaries = cu_seqlens.to(device="cpu").tolist()
    states = torch.zeros(
        batch, state_len, dim, dtype=x.dtype, device=x.device
    ).transpose(1, 2)

    for i in range(batch):
        end = boundaries[i + 1]
        start = max(boundaries[i], end - state_len)
        n = end - start
        if n > 0:
            states[i, :, state_len - n :] = x[start:end].transpose(0, 1)
    return states
