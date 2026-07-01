"""Python entry point for the repro kernel. dtype/dim/batch are all
comptime-fixed in kernel.mojo, so this only takes two GPU addresses."""

from std.gpu.host import DeviceContext
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from kernel import dtype, kBatch, kNThreadsUpdate, update_kernel


def causal_conv1d_update_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var x_addr = Int(py=args[0])
    var o_addr = Int(py=args[1])

    var ctx = DeviceContext()

    var grid = (kBatch,)

    var x_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=x_addr
    )
    var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=o_addr
    )

    var compiled = ctx.compile_function[update_kernel]()
    ctx.enqueue_function(
        compiled,
        x_ptr,
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
        return m.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
