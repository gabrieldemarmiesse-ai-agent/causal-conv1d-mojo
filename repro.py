"""Minimal repro: causal_conv1d_update on Apple Metal (mps) returns
all-zero output the very first time a fresh mojo build's kernel is
dispatched, when `~/.cache/modular`'s Mojo transform cache is cold.

    mojo --clear-cache -f
    uv run python repro.py

FAILS every time on a cold cache; passes if you run it again (cache now
warm) or call the same kernel a second time in one process.
"""

import torch

import causal_conv1d_mojo

torch.manual_seed(0)
device = "mps"
dtype = torch.float16
width = 4
B, D = 2, 16
state_len = width - 1
x = torch.randn(B, D, dtype=dtype, device=device)
weight = torch.randn(D, width, dtype=dtype, device=device)
bias = torch.randn(D, dtype=dtype, device=device)
state = torch.randn(B, D, state_len, dtype=dtype, device=device)

out = causal_conv1d_mojo.causal_conv1d_update(x, state, weight, bias)

print("out:", out)
max_abs = out.abs().max().item()
print("max_abs =", max_abs)
assert max_abs > 0, "FAILED: kernel wrote all zeros (cold mojo/modular cache)"
print("PASSED")
