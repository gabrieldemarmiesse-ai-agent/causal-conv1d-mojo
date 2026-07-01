"""GPU kernel for the repro: copies x into out, one element per thread.
The actual computation is irrelevant to the bug (a fresh-Metal-dispatch
correctness issue), so this has been trimmed down from causal_conv1d's
real update step to the simplest op that still writes non-trivial data."""

from std.gpu import block_idx, thread_idx


comptime kNThreadsUpdate: Int = 16
comptime dtype = DType.float16


def update_kernel(
    dim: Int32,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    # kNThreadsUpdate == dim (16, the repro's fixed D) exactly, so a
    # single block dimension covers the whole channel range — no
    # block_idx.y and no bounds check needed.
    var batch_id: Int32 = Int32(block_idx.x)
    var channel_id: Int32 = Int32(thread_idx.x)

    # x/out both have shape (batch, dim) and are always freshly allocated
    # + contiguous in this repro, so batch stride = dim and channel
    # stride = 1 — no need to pass strides at all.
    var lane_offset = Int(batch_id * dim + channel_id)
    o_ptr[lane_offset] = x_ptr[lane_offset]
