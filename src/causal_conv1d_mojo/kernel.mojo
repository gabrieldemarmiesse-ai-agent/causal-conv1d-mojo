"""GPU kernel for the repro: copies x into out, one element per thread.
The actual computation is irrelevant to the bug (a fresh-Metal-dispatch
correctness issue), so this has been trimmed down from causal_conv1d's
real update step to the simplest op that still writes non-trivial data."""

from std.gpu import block_idx, thread_idx


comptime kNThreadsUpdate: Int = 16
comptime kBatch: Int = 2
comptime dtype = DType.float16


def update_kernel(
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    # kNThreadsUpdate == dim (16, the repro's fixed D) exactly, so a
    # single block dimension covers the whole channel range — no
    # block_idx.y and no bounds check needed. dim is comptime-fixed too
    # (the repro always uses D=16), so it never needs to cross the
    # Python/Mojo boundary as a runtime value.
    var batch_id = Int(block_idx.x)
    var channel_id = Int(thread_idx.x)

    # x/out both have shape (batch, dim) and are always freshly allocated
    # + contiguous in this repro, so batch stride = dim and channel
    # stride = 1 — no need to pass strides at all.
    var lane_offset = batch_id * kNThreadsUpdate + channel_id
    o_ptr[lane_offset] = x_ptr[lane_offset]
