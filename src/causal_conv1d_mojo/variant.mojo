"""Static variant entry point for causal_conv1d_update (trimmed repro:
dtype=float16, width=4 hardcoded in kernel.mojo, no -D defines needed).
"""

from std.gpu.host import DeviceContext
from std.gpu.host.device_context import _DeviceContextPtr, _DeviceContextCpp
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from kernel import dtype, kNThreadsUpdate, update_kernel


def causal_conv1d_update_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Create + retain a process-lifetime DeviceContext, and leak the
    wrapper. Called once from the Python side; the returned address is
    reused for every subsequent `causal_conv1d_update_variant` call.
    """
    var ctx = DeviceContext()
    ctx._retain()
    var addr: Int = Int(ctx._handle.value())
    return PythonObject(addr)


def causal_conv1d_update_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var x_addr = Int(py=args[0])
    var w_addr = Int(py=args[1])
    var state_addr = Int(py=args[2])
    var o_addr = Int(py=args[3])
    var batch_int = Int(py=args[4])
    var dim_int = Int(py=args[5])
    # ctx_handle is appended as args[6] by call_update.
    var ctx_handle_addr = Int(py=args[6])

    # Reconstruct a non-owning DeviceContext from the cached handle —
    # avoids creating a fresh DeviceContext (and its stream) every call.
    var raw_ctx_ptr = UnsafePointer[_DeviceContextCpp, MutUntrackedOrigin](
        unsafe_from_address=ctx_handle_addr
    )
    var ctx = DeviceContext(_DeviceContextPtr[mut=True](raw_ctx_ptr))

    var grid = (batch_int,)

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
    return PythonObject(None)


@export
def PyInit_variant() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("variant")
        m.def_py_function[causal_conv1d_update_variant](
            "causal_conv1d_update_variant"
        )
        m.def_py_function[causal_conv1d_update_acquire_ctx](
            "causal_conv1d_update_acquire_ctx"
        )
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
