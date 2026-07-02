"""Pure-Mojo (no Python, no PyTorch) deterministic repro of the
"first Mojo dispatch reads/writes zero on Apple Metal" bug, in the exact
shape that bites real PyTorch-interop code.

A Mojo kernel is dispatched against a foreign MTLBuffer sub-allocated
from a hazard-TRACKED MTLHeap (precisely how PyTorch's MPS allocator
allocates every tensor: tracked shared heaps, see MPSAllocator.mm), with
the buffer referenced only by its raw `gpuAddress` via
`UnsafePointer(unsafe_from_address=...)`.

    uv run mojo run repro_heap.mojo      # FAILS, 100% deterministic

Behavior matrix (edit the two comptime toggles to verify):

    kPriorGpuTouch  kSleepSeconds  result
    False           0              FAIL  (heap never made resident)
    True            0              PASS  (heap still resident from touch)
    True            2.0 (default)  FAIL  (macOS evicts idle GPU memory
                                          after ~1-1.5s; this is the
                                          PyTorch-interop case)

Root cause (confirmed by disassembling libAsyncRTMojoBindings.dylib):
`MetalDeviceContext::enqueueFunctionExecDirect` calls
[MTLComputeCommandEncoder useResource:usage:] ONLY for addresses found
in Mojo's internal allocation-tracking table (populated by Mojo's own
buffer allocator). A foreign buffer referenced by raw gpuAddress is
never in that table, so it is never declared to the encoder. The macOS
AGX driver evicts idle GPU memory after ~1s (same eviction documented in
ggml-org/llama.cpp#10119); an undeclared, evicted allocation is then
silently read as zeros and writes to it are dropped -- no Metal API
validation error, no command-buffer error.
"""

from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.time import sleep


comptime dtype = DType.float16
comptime AnyPtr = UnsafePointer[NoneType, MutAnyOrigin]

comptime kPriorGpuTouch: Bool = True
comptime kSleepSeconds: Float64 = 2.0


def msg_send(
    receiver: AnyPtr,
    selector: AnyPtr,
    a0: UInt = 0,
    a1: UInt = 0,
    a2: UInt = 0,
    a3: UInt = 0,
) -> AnyPtr:
    # Every call site must share one exact `objc_msgSend` signature — the
    # compiler treats differently-shaped external_call[] invocations of
    # the same symbol as conflicting extern declarations. Unused trailing
    # args are harmless: the callee only reads the registers it expects.
    return external_call["objc_msgSend", AnyPtr](receiver, selector, a0, a1, a2, a3)


def sel(name: StaticString) -> AnyPtr:
    return external_call["sel_registerName", AnyPtr](name.unsafe_ptr())


def cls(name: StaticString) -> AnyPtr:
    return external_call["objc_getClass", AnyPtr](name.unsafe_ptr())


def copy_kernel(
    x_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    o_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    o_ptr[0] = x_ptr[0]


def main() raises:
    var device = external_call["MTLCreateSystemDefaultDevice", AnyPtr]()

    # A hazard-TRACKED, shared-storage MTLHeap — the same configuration
    # PyTorch's MPS allocator uses for its tensor pools.
    var desc = msg_send(msg_send(cls("MTLHeapDescriptor"), sel("alloc")), sel("init"))
    _ = msg_send(desc, sel("setSize:"), 1 << 20)
    _ = msg_send(desc, sel("setStorageMode:"), 0)  # Shared
    _ = msg_send(desc, sel("setHazardTrackingMode:"), 2)  # Tracked

    var heap = msg_send(device, sel("newHeapWithDescriptor:"), UInt(Int(desc)))

    var sel_new_buffer = sel("newBufferWithLength:options:")
    var x_buf = msg_send(heap, sel_new_buffer, 2, 0)
    var o_buf = msg_send(heap, sel_new_buffer, 2, 0)

    var sel_contents = sel("contents")
    var x_host = msg_send(x_buf, sel_contents).bitcast[Scalar[dtype]]()
    x_host[0] = Scalar[dtype](42.0)
    var o_host = msg_send(o_buf, sel_contents).bitcast[Scalar[dtype]]()
    o_host[0] = Scalar[dtype](99.0)  # sentinel: distinguishes "dispatch
    # lost entirely" (stays 99) from "ran but read zeros" (becomes 0)

    if kPriorGpuTouch:
        # Mimic PyTorch's role: submit real GPU work touching this heap
        # (a blit filling a third scratch buffer from the same heap) on a
        # separate command queue. This makes the heap resident.
        var t_buf = msg_send(heap, sel_new_buffer, 2, 0)
        var queue = msg_send(device, sel("newCommandQueue"))
        var cb = msg_send(queue, sel("commandBuffer"))
        var blit = msg_send(cb, sel("blitCommandEncoder"))
        # fillBuffer:range:value: — NSRange passed as two register args
        _ = msg_send(blit, sel("fillBuffer:range:value:"), UInt(Int(t_buf)), 0, 2, 7)
        _ = msg_send(blit, sel("endEncoding"))
        _ = msg_send(cb, sel("commit"))
        _ = msg_send(cb, sel("waitUntilCompleted"))

    if kSleepSeconds > 0.0:
        # Idle the GPU past the driver's ~1s eviction window — the same
        # delay a cold-cache `mojo build` (or any think-time gap between
        # calls) introduces in real interop code.
        sleep(kSleepSeconds)

    var x_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(msg_send(x_buf, sel("gpuAddress")))
    )
    var o_ptr = UnsafePointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(msg_send(o_buf, sel("gpuAddress")))
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

    print("out:", o_host[0])
    if o_host[0] == Scalar[dtype](42.0):
        print("PASSED")
    elif o_host[0] == Scalar[dtype](99.0):
        print("FAILED: dispatch lost entirely (sentinel untouched)")
    else:
        print("FAILED: kernel ran but read the idle heap as zero")
