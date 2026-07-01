"""Minimal repro: causal_conv1d_update on Apple Metal (mps) returns
all-zero output the very first time a fresh mojo build's kernel is
dispatched, when `~/.cache/modular`'s Mojo transform cache is cold.

    rm -rf ~/.cache/causal_conv1d_mojo && mojo --clear-cache -f
    uv run python repro.py

FAILS every time on a cold cache; passes if you run it again (cache now
warm) or call the same kernel a second time in one process.
"""

import torch

import causal_conv1d_mojo
from causal_conv1d_mojo import causal_conv1d_update_ref


def _max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


torch.manual_seed(0)
device = "mps"
activation = "silu"
dtype = torch.float16
width = 4
B, D = 2, 16
state_len = width - 1
x = torch.randn(B, D, dtype=dtype, device=device)
weight = torch.randn(D, width, dtype=dtype, device=device)
bias = torch.randn(D, dtype=dtype, device=device)
state = torch.randn(B, D, state_len, dtype=dtype, device=device)

state_ours = state.clone()
state_ref = state.clone()

out_ours = causal_conv1d_mojo.causal_conv1d_update(
    x, state_ours, weight, bias=bias, activation=activation
)
out_ref = causal_conv1d_update_ref(
    x, state_ref, weight, bias=bias, activation=activation
)

print("out_ours:", out_ours)
print("out_ref :", out_ref)
diff = _max_diff(out_ours, out_ref)
print("max_diff =", diff)
assert diff < 0.02, f"FAILED: max_diff={diff}"
print("PASSED")
