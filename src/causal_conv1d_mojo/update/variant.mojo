"""Static variant entry point for causal_conv1d_update (trimmed repro:
dtype=float16, width=4 hardcoded in kernel.mojo, no -D defines needed).
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from launch import launch_update
from _ctx import acquire_ctx_handle


def causal_conv1d_update_acquire_ctx(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    """Create + retain a process-lifetime DeviceContext.

    Called once per variant from the Python side; the returned address
    is reused for every subsequent `causal_conv1d_update_variant` call.
    """
    var addr: Int = acquire_ctx_handle()
    return PythonObject(addr)


def causal_conv1d_update_variant(
    mut py_self: PythonObject,
    mut args: PythonObject,
) raises -> PythonObject:
    var x_addr = Int(py=args[0])
    var w_addr = Int(py=args[1])
    var b_addr = Int(py=args[2])
    var state_addr = Int(py=args[3])
    var o_addr = Int(py=args[4])
    var batch_int = Int(py=args[5])
    var dim_int = Int(py=args[6])
    var seqlen_int = Int(py=args[7])
    var state_len_int = Int(py=args[8])
    var x_b_stride = Int32(py=args[9])
    var x_c_stride = Int32(py=args[10])
    var x_l_stride = Int32(py=args[11])
    var w_c_stride = Int32(py=args[12])
    var state_b_stride = Int32(py=args[13])
    var state_c_stride = Int32(py=args[14])
    var state_l_stride = Int32(py=args[15])
    var o_b_stride = Int32(py=args[16])
    var o_c_stride = Int32(py=args[17])
    var o_l_stride = Int32(py=args[18])
    # ctx_handle is appended as args[19] by call_update.
    var ctx_handle_addr = Int(py=args[19])

    if batch_int == 0 or dim_int == 0:
        return PythonObject(None)

    launch_update(
        batch_int,
        dim_int,
        seqlen_int,
        state_len_int,
        x_addr,
        w_addr,
        b_addr,
        state_addr,
        o_addr,
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
