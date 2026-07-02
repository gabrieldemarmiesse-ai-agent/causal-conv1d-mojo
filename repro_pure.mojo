"""Pure-Mojo (no Python, no PyTorch) repro, variant 2 of 2: a Mojo kernel
launched against a foreign *device-allocated* MTLBuffer with
MTLResourceHazardTrackingModeUntracked reads/writes zero, 100%
deterministically, on any cache state.

This is a sibling manifestation of the same root cause as
`repro_heap.mojo` (which matches the real PyTorch-interop case — go read
that one first): Mojo's Metal backend only emits
[encoder useResource:usage:] for buffers found in its own internal
allocation table, so foreign buffers referenced by raw gpuAddress are
never declared to the encoder. Empirically on macOS 26 / M4:

  - foreign device-allocated buffer, TRACKED   -> always works (the
    driver appears to keep plain device allocations resident)
  - foreign device-allocated buffer, UNTRACKED -> always fails (this file)
  - foreign HEAP sub-allocation, TRACKED       -> works only within ~1s
    of the last GPU work touching that heap (repro_heap.mojo; this is
    what PyTorch MPS tensors are)

Toggle `kUntrackedShared` to 0 (tracked/default) below to see the same
dispatch always succeed instead.

    uv run mojo run repro_pure.mojo
"""

from std.ffi import external_call
from std.gpu.host import DeviceContext


comptime dtype = DType.float16
comptime AnyPtr = UnsafePointer[NoneType, MutAnyOrigin]

# MTLResourceStorageModeShared (0) | MTLResourceHazardTrackingModeUntracked
# (1 << 8 = 256).
comptime kUntrackedShared: UInt = 256


def msg_send(receiver: AnyPtr, selector: AnyPtr, a0: UInt = 0, a1: UInt = 0) -> AnyPtr:
    # Every call site must share one exact `objc_msgSend` signature — the
    # compiler treats differently-shaped external_call[] invocations of
    # the same symbol as conflicting extern declarations. Unused trailing
    # args are harmless: the real method implementation only reads the
    # registers it actually expects.
    return external_call["objc_msgSend", AnyPtr](receiver, selector, a0, a1)


def sel(name: StaticString) -> AnyPtr:
    return external_call["sel_registerName", AnyPtr](name.unsafe_ptr())


def copy_kernel(
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    o_ptr[0] = x_ptr[0]


def main() raises:
    var device = external_call["MTLCreateSystemDefaultDevice", AnyPtr]()

    var sel_new_buffer = sel("newBufferWithLength:options:")
    var x_buf = msg_send(device, sel_new_buffer, 2, kUntrackedShared)
    var o_buf = msg_send(device, sel_new_buffer, 2, kUntrackedShared)

    var sel_gpu_address = sel("gpuAddress")
    var x_gpu_addr = Int(msg_send(x_buf, sel_gpu_address))
    var o_gpu_addr = Int(msg_send(o_buf, sel_gpu_address))

    var sel_contents = sel("contents")
    var x_host = msg_send(x_buf, sel_contents).bitcast[Scalar[dtype]]()
    x_host[0] = Scalar[dtype](42.0)

    var x_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=x_gpu_addr
    )
    var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=o_gpu_addr
    )

    var ctx = DeviceContext()
    var compiled = ctx.compile_function[copy_kernel]()
    ctx.enqueue_function(
        compiled,
        x_ptr,
        o_ptr,
        grid_dim=(1,),
        block_dim=(1,),
    )
    ctx.synchronize()

    var o_host = msg_send(o_buf, sel_contents).bitcast[Scalar[dtype]]()
    print("out:", o_host[0])
    if o_host[0] == Scalar[dtype](0.0):
        print("FAILED: kernel wrote/read zero (untracked hazard mode)")
    else:
        print("PASSED")
