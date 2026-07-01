"""GPU single-step update kernel for causal_conv1d (trimmed repro:
linear/non-circular state only, no state_indices)."""

from std.gpu import block_idx, thread_idx
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA
from std.utils.index import StaticTuple


comptime kNThreadsUpdate: Int = 64
comptime dtype = DType.float16


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kNThreadsUpdate))
)
def update_kernel(
    dim: Int32,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    state_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    comptime accum_t = DType.float32

    var batch_id: Int32 = Int32(block_idx.x)
    var channel_id: Int32 = Int32(block_idx.y) * Int32(kNThreadsUpdate) + Int32(
        thread_idx.x
    )
    if channel_id >= dim:
        return

    # x/state/out all have shape (batch, dim, 1) and are always freshly
    # allocated + contiguous in this repro, so batch stride = dim and
    # channel stride = 1 — no need to pass strides at all.
    var lane_offset = Int(batch_id * dim + channel_id)
    var x_lane = x_ptr + lane_offset
    var out_lane = o_ptr + lane_offset
    var state_lane = state_ptr + lane_offset
    # weight has shape (dim, 2), also contiguous — channel stride = 2.
    var w_lane = w_ptr + Int(channel_id * 2)

    var w0: Scalar[accum_t] = w_lane[0].cast[accum_t]()
    var w1: Scalar[accum_t] = w_lane[1].cast[accum_t]()

    # Phase 2: read the single history value (state_len is always
    # width-1=1 for this repro, so there's exactly one).
    var x0: Scalar[accum_t] = state_lane[0].cast[accum_t]()

    # Phase 3: consume the single new x, write into state, emit output.
    var x_val = x_lane[0]
    state_lane[0] = x_val
    var x1: Scalar[accum_t] = x_val.cast[accum_t]()

    var out_val: Scalar[accum_t] = w0 * x0 + w1 * x1

    out_lane[0] = out_val.cast[dtype]()
