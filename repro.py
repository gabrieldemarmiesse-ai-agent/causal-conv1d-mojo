"""Minimal repro (no pytest/beartype): run

    rm -rf ~/.cache/causal_conv1d_mojo && mojo --clear-cache -f
    uv run python repro.py

FAILS every time on a cold cache; passes if the mojo/modular cache is
already warm.
"""

import torch

import causal_conv1d_mojo
from causal_conv1d_mojo import causal_conv1d_update_ref

torch.manual_seed(0)

B, D, width = 2, 16, 4
state_len = width - 1
device, dtype = "mps", torch.float16

x = torch.randn(B, D, dtype=dtype, device=device)
weight = torch.randn(D, width, dtype=dtype, device=device)
bias = torch.randn(D, dtype=dtype, device=device)
state = torch.randn(B, D, state_len, dtype=dtype, device=device)

state_ours = state.clone()
state_ref = state.clone()

out_ours = causal_conv1d_mojo.causal_conv1d_update(
    x, state_ours, weight, bias=bias, activation="silu"
)
out_ref = causal_conv1d_update_ref(
    x, state_ref, weight, bias=bias, activation="silu"
)

max_diff = (out_ours.float() - out_ref.float()).abs().max().item()
print("out_ours:", out_ours)
print("out_ref :", out_ref)
print("max_diff =", max_diff)
assert max_diff < 0.02, f"FAILED: max_diff={max_diff}"
print("PASSED")
