"""GPU kernel for the repro: copies x into out, one element per thread.
The actual computation is irrelevant to the bug (a fresh-Metal-dispatch
correctness issue), so this has been trimmed down from causal_conv1d's
real update step to the simplest op that still writes non-trivial data."""

from std.gpu import thread_idx


comptime kNThreadsUpdate: Int = 32
comptime dtype = DType.float16


def update_kernel(
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    # x/out are (batch=2, dim=16) contiguous and always freshly allocated
    # in this repro, so a single 32-thread block can just copy the whole
    # flat buffer — no grid, no per-element index math needed.
    var i = Int(thread_idx.x)
    o_ptr[i] = x_ptr[i]
