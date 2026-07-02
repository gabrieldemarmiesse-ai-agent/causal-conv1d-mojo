"""The same flow as repro.py, plus the validated library-side workaround:
immediately before the Mojo dispatch, "revive" every argument tensor's
MTLHeap with a tiny GPU no-op, then synchronize. The subsequent Mojo
dispatch then lands within the driver's ~1s residency window.

    mojo --clear-cache -f
    uv run python workaround.py     # PASSES, even on a cold cache

Why per-tensor: PyTorch's MPS allocator uses multiple heaps (an 8 MiB
small-object pool, 32 MiB+ large pools), and residency is per-heap —
reviving one tensor's heap does not revive another's. (With only the
source tensor's heap left idle, the kernel *runs* but reads it as zeros,
which is even nastier than the fully-lost dispatch.)

The in-place `add_(0)` is numerically a no-op; for tensors that require
grad, use a read-only touch (e.g. `t.view(-1)[0:1].clone()`) instead to
avoid bumping the autograd version counter.
"""

import time

import torch

from causal_conv1d_mojo import _get_variant_fn, gpu_address

torch.manual_seed(0)
device = "mps"
dtype = torch.float16
x = torch.randn(1, dtype=dtype, device=device)
out = torch.empty_like(x)

variant_fn = _get_variant_fn()  # cold cache: several seconds of mojo build
time.sleep(2)  # extra GPU idle time on top, for good measure

# --- WORKAROUND: revive every argument tensor's heap, then sync ---
for t in (x, out):
    t.view(-1)[0:1].add_(0)
torch.mps.synchronize()
# ------------------------------------------------------------------

variant_fn(gpu_address(x), gpu_address(out))

print("x:  ", x)
print("out:", out)
assert out.item() == x.item(), "workaround failed"
print("PASSED")
