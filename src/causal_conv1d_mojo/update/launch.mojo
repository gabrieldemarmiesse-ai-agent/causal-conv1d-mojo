"""Single-variant launch helper for the GPU update kernel (trimmed
repro; Metal-only, no state_indices/cache_seqlens/external stream)."""

from std.gpu.host import DeviceContext
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.math import ceildiv

from kernel import dtype, kNThreadsUpdate, update_kernel


def launch_update(
    batch_int: Int,
    dim_int: Int,
    seqlen_int: Int,
    state_len_int: Int,
    x_addr: Int,
    w_addr: Int,
    b_addr: Int,
    state_addr: Int,
    o_addr: Int,
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
    ctx_handle_addr: Int,
) raises:
    # Reconstruct a non-owning DeviceContext from the cached handle —
    # avoids creating a fresh DeviceContext (and its stream) every call.
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutUntrackedOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))

    var grid = (batch_int, ceildiv(dim_int, kNThreadsUpdate))

    var x_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=x_addr
    )
    var w_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=w_addr
    )
    var b_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=b_addr
    )
    var state_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=state_addr
    )
    var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=o_addr
    )

    var compiled = ctx.compile_function[update_kernel]()
    ctx.enqueue_function(
        compiled,
        Int32(dim_int),
        Int32(seqlen_int),
        Int32(state_len_int),
        x_ptr,
        w_ptr,
        b_ptr,
        state_ptr,
        o_ptr,
        x_b_stride,
        x_c_stride,
        x_l_stride,
        w_c_stride,
        state_b_stride,
        state_c_stride,
        state_l_stride,
        o_b_stride,
        o_c_stride,
        o_l_stride,
        grid_dim=grid,
        block_dim=(kNThreadsUpdate,),
    )
    ctx.synchronize()
