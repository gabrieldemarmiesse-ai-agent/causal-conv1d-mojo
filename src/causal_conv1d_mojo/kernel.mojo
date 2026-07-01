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
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    state_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_b_stride: Int32,
    x_c_stride: Int32,
    w_c_stride: Int32,
    state_b_stride: Int32,
    state_c_stride: Int32,
    o_b_stride: Int32,
    o_c_stride: Int32,
):
    comptime accum_t = DType.float32

    var batch_id: Int32 = Int32(block_idx.x)
    var channel_id: Int32 = Int32(block_idx.y) * Int32(kNThreadsUpdate) + Int32(
        thread_idx.x
    )
    if channel_id >= dim:
        return

    var x_lane = x_ptr + Int(batch_id * x_b_stride + channel_id * x_c_stride)
    var out_lane = o_ptr + Int(batch_id * o_b_stride + channel_id * o_c_stride)
    var state_lane = state_ptr + Int(
        batch_id * state_b_stride + channel_id * state_c_stride
    )
    var w_lane = w_ptr + Int(channel_id * w_c_stride)

    var w0: Scalar[accum_t] = w_lane[0].cast[accum_t]()
    var w1: Scalar[accum_t] = w_lane[1].cast[accum_t]()
    var bias_v: Scalar[accum_t] = bias_ptr[Int(channel_id)].cast[accum_t]()

    # Phase 2: read the single history value (state_len is always
    # width-1=1 for this repro, so there's exactly one).
    var x0: Scalar[accum_t] = state_lane[0].cast[accum_t]()

    # Phase 3: consume the single new x, write into state, emit output.
    var x_val = x_lane[0]
    state_lane[0] = x_val
    var x1: Scalar[accum_t] = x_val.cast[accum_t]()

    var out_val: Scalar[accum_t] = bias_v + w0 * x0 + w1 * x1

    out_lane[0] = out_val.cast[dtype]()
