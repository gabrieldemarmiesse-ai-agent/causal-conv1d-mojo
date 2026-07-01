"""Static variant entry point for causal_conv1d_update (trimmed repro:
dtype=float16, width=4 hardcoded in kernel.mojo, no -D defines needed).
"""

from std.gpu.host import DeviceContext
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from launch import launch_update


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

    launch_update(
        batch_int,
        dim_int,
        x_addr,
        w_addr,
        state_addr,
        o_addr,
        ctx_handle_addr,
    )
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
