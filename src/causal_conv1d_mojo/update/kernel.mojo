"""GPU single-step update kernel for causal_conv1d (trimmed repro:
linear/non-circular state only, no state_indices)."""

from std.gpu import block_idx, thread_idx
from std.gpu.globals import MAX_THREADS_PER_BLOCK_METADATA
from std.sys import size_of
from std.utils.index import StaticTuple

from _silu import _silu_f32


comptime kNThreadsUpdate: Int = 64


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(kNThreadsUpdate))
)
def update_kernel[
    dtype: DType,
    width: Int,
    has_bias: Bool,
    apply_silu: Bool,
](
    dim: Int32,
    seqlen: Int32,
    state_len: Int32,
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    w_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    state_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_b_stride: Int32,
    x_c_stride: Int32,
    x_l_stride: Int32,
    w_c_stride: Int32,
    state_b_stride: Int32,
    state_c_stride: Int32,
    state_l_stride: Int32,
    o_b_stride: Int32,
    o_c_stride: Int32,
    o_l_stride: Int32,
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

    var weights = SIMD[accum_t, width](0)
    var w_vec = w_lane.load[width=width, alignment = size_of[dtype]() * width](0)
    comptime for k in range(width):
        weights[k] = w_vec[k].cast[accum_t]()

    var bias_v: Scalar[accum_t] = 0
    comptime if has_bias:
        bias_v = bias_ptr[Int(channel_id)].cast[accum_t]()

    var sl: Int32 = state_len
    var advance_len: Int32 = seqlen
    var x_vals = SIMD[accum_t, width](0)

    # Phase 1: shift state left by `seqlen`.
    var n_shift: Int32 = sl - advance_len - Int32(width - 1)
    if n_shift > 0:
        var sj: Int32 = 0
        while True:
            state_lane[Int(sj * state_l_stride)] = state_lane[
                Int((sj + advance_len) * state_l_stride)
            ]
            sj = sj + Int32(1)
            if not (sj < n_shift):
                break

    # Phase 2: read trailing W-1 history into x_vals (with writeback for
    # the small-state_len edge case).
    var state_vals = SIMD[dtype, width - 1](0)
    comptime if width == 3 or width == 4:
        var s_vec = state_lane.load[width = width - 1, alignment=2](
            Int((sl - Int32(width - 1)) * state_l_stride)
        )
        comptime for i in range(width - 1):
            state_vals[i] = s_vec[i]
    else:
        comptime for i in range(width - 1):
            var read_idx: Int32 = sl - Int32(width - 1) + Int32(i)
            state_vals[i] = state_lane[Int(read_idx * state_l_stride)]

    comptime for i in range(width - 1):
        var write_idx: Int32 = sl - advance_len - Int32(width - 1) + Int32(i)
        if Int32(i) < advance_len + Int32(width - 1) and write_idx >= 0:
            state_lane[Int(write_idx * state_l_stride)] = state_vals[i]
        x_vals[i] = state_vals[i].cast[accum_t]()

    # Phase 3: walk new x, write into state, emit output.
    if advance_len < 1:
        return
    var i: Int32 = 0
    while True:
        var x_val = x_lane[Int(i * x_l_stride)]

        var write_idx: Int32 = sl - advance_len + i
        if i < advance_len and write_idx >= 0:
            state_lane[Int(write_idx * state_l_stride)] = x_val

        x_vals[width - 1] = x_val.cast[accum_t]()

        var out_val: Scalar[accum_t] = bias_v

        comptime for k in range(width):
            out_val += weights[k] * x_vals[k]

        comptime if apply_silu:
            out_val = _silu_f32(Float32(out_val))

        out_lane[Int(i * o_l_stride)] = out_val.cast[dtype]()

        comptime for k in range(width - 1):
            x_vals[k] = x_vals[k + 1]

        i = i + Int32(1)
        if not (i < advance_len):
            break
