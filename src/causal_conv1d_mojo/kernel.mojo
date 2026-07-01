"""GPU kernel for the repro: copies x into out, one element per thread.
The actual computation is irrelevant to the bug (a fresh-Metal-dispatch
correctness issue), so this has been trimmed down from causal_conv1d's
real update step to the simplest op that still writes non-trivial data."""

comptime kNThreadsUpdate: Int = 1
comptime dtype = DType.float16


def update_kernel(
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    # x/out are single-element buffers in this repro, so one thread
    # copying one element is enough to trigger the bug.
    o_ptr[0] = x_ptr[0]
