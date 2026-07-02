"""Minimal PyTorch-interop repro: a Mojo kernel dispatched against torch
MPS tensors (referenced by raw gpuAddress) silently does nothing, and the
output tensor stays all-zero.

    mojo --clear-cache -f
    uv run python repro.py

FAILS every time on a cold cache; passes if you run it again (cache now
warm) or call the same kernel a second time in one process.

NOTE the cache itself is a red herring — it is only a *delay amplifier*.
The real trigger (see BUG_REPORT.md) is >~1-1.5s of GPU idle time between
the last GPU work that touched the tensors' MTLHeap (torch's randn here)
and the Mojo dispatch: macOS evicts idle GPU memory after ~1s, and Mojo
never re-declares foreign buffers to its compute encoder
(useResource:usage: is only emitted for Mojo's own allocations). A cold
`mojo build` takes several seconds (always fails); a warm one ~1.2s
(borderline, passes); a second in-process call has a ~0s gap (always
passes — the "self-healing"). `repro_heap.mojo` reproduces the same
failure deterministically in pure Mojo with no Python/PyTorch at all,
and `workaround.py` shows the library-side mitigation.
"""

import torch

import causal_conv1d_mojo

torch.manual_seed(0)
device = "mps"
dtype = torch.float16
x = torch.randn(1, dtype=dtype, device=device)

out = causal_conv1d_mojo.causal_conv1d_update(x)

print("out:", out)
max_abs = out.abs().max().item()
print("max_abs =", max_abs)
assert max_abs > 0, "FAILED: kernel wrote all zeros (cold mojo/modular cache)"
print("PASSED")
