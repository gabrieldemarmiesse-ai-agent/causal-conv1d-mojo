"""Single-variant launch helper for the GPU update kernel (trimmed
repro; Metal-only, no state_indices/cache_seqlens/external stream)."""

from std.gpu.host import DeviceContext
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.math import ceildiv

from kernel import dtype, kNThreadsUpdate, update_kernel


def launch_update(
    batch_int: Int,
    dim_int: Int,
    x_addr: Int,
    w_addr: Int,
    state_addr: Int,
    o_addr: Int,
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
        x_ptr,
        w_ptr,
        state_ptr,
        o_ptr,
        grid_dim=grid,
        block_dim=(kNThreadsUpdate,),
    )
    ctx.synchronize()
