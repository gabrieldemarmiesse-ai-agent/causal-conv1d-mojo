"""Pure-pytorch reference for the two public APIs:

  - `causal_conv1d_ref`        — pytorch fallback for `causal_conv1d_fn`
  - `causal_conv1d_update_ref` — pytorch fallback for `causal_conv1d_update`

Both are exposed at the top level (`from causal_conv1d_mojo import
causal_conv1d_ref`) so callers can validate their own kernels or use
the pure-pytorch path on hardware where the Mojo kernels don't apply.

Implementations are copied verbatim from upstream `causal_conv1d`
(Tri Dao, 2024 — BSD-3-Clause); inlining them here removes the runtime
dependency on the upstream C++ extension (multi-minute source build).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = None,
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    With seq_idx and initial_states together, the virtual initial-state
    positions carry seq_idx[:, 0], so only the first packed sequence in
    each batch row can read them. Negative seq_idx positions are padding
    and produce zero output.

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x_input = x
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if seq_idx is not None:
        assert seq_idx.shape == (x.shape[0], seqlen)
        assert not return_final_states, "seq_idx does not support return_final_states"
        if initial_states is None:
            x = F.pad(x, (width - 1, 0))
            initial_state_seq_idx = seq_idx.new_full((x.shape[0], width - 1), -1)
        else:
            x = torch.cat([initial_states, x], dim=-1)
            initial_state_seq_idx = seq_idx[:, :1].expand(-1, width - 1)
        seq_idx_with_state = torch.cat([initial_state_seq_idx, seq_idx], dim=-1)
        x_windows = x.unfold(dimension=-1, size=width, step=1)
        seq_idx_windows = seq_idx_with_state.unfold(dimension=-1, size=width, step=1)
        same_sequence = seq_idx_windows == seq_idx.unsqueeze(-1)
        out = torch.sum(
            x_windows * same_sequence.unsqueeze(1) * weight.view(1, dim, 1, width),
            dim=-1,
        )
        if bias is not None:
            out = out + bias.view(1, dim, 1)
        out = out.masked_fill((seq_idx < 0).unsqueeze(1), 0.0)
    elif initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        # The kernel converts both input/state values to fp32 only after
        # loading them from their shared input dtype. Cast the state to
        # weight dtype here before concatenation: torch.cat promotes an
        # fp16/bf16 pair to fp32, which would otherwise make F.conv1d
        # reject fp16/bf16 weights and bias for that one mixed pair.
        x = torch.cat([initial_states.to(weight.dtype), x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        # (batch, dim, width - 1)
        # final_states is input_t upstream and copies the original input,
        # not the weight_t cast used only for convolution arithmetic.
        final_states = F.pad(x_input, (width - 1 - x_input.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    cache_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index `cache_seqlens % state_len` before
        performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        # (batch, dim, state_len + seqlen)
        x_new_input = torch.cat([conv_state, x], dim=-1)
        # State is input_t upstream: shifting it must not round the
        # retained history through weight_t when the dtypes differ.
        conv_state.copy_(x_new_input[:, :, -state_len:])
        x_new = x_new_input.to(weight.dtype)
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    # F.conv1d(groups=dim, padding=0) crashes on AMD ROCm gfx942 (MI300A) even
    # with MIOPEN_DISABLE_CACHE=1 — MIOpen's unpadded grouped conv path is broken.
    # The padded path used in causal_conv1d_ref above is fine; only padding=0 hits
    # the bad code path. unfold+mul+sum is numerically identical and backend-agnostic.
    x_unfolded = x_new.unfold(-1, width, 1)  # (batch, dim, L-width+1, width)
    out = (x_unfolded * weight.unsqueeze(0).unsqueeze(2)).sum(-1)[:, :, -seqlen:]
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)
