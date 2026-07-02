# Mojo/Metal: kernels silently read/write zero for foreign GPU buffers — `useResource:` is only emitted for Mojo's own allocations

## Summary

On Apple Metal, a Mojo kernel launched via `DeviceContext.enqueue_function`
against a GPU buffer that Mojo did not allocate itself (referenced by raw
address via `UnsafePointer(unsafe_from_address=...)`) silently reads zeros
from it and/or drops writes to it, with no error anywhere — whenever the
Metal driver does not happen to have that allocation resident.

**Root cause** (confirmed by disassembling `libAsyncRTMojoBindings.dylib`,
see "Evidence 4"): `MetalDeviceContext::enqueueFunctionExecDirect` calls
`[MTLComputeCommandEncoder useResource:usage:]` **only for addresses found
in Mojo's internal allocation-tracking table**, which is populated only by
Mojo's own buffer allocator. Foreign buffers are never declared to the
encoder, so their residency is left to chance:

| foreign buffer kind                          | result                                    |
|----------------------------------------------|-------------------------------------------|
| device-allocated, hazard-**tracked**          | works (driver keeps it resident, empirically even after 15 s idle) |
| device-allocated, hazard-**untracked**        | **always fails** (`repro_pure.mojo`)       |
| **MTLHeap sub-allocation**, hazard-tracked    | works only within **~1–1.5 s** of the last GPU work that touched that heap; fails after (`repro_heap.mojo`) |

The third row is the one that bites real interop code: **PyTorch's MPS
allocator sub-allocates every tensor from hazard-tracked shared `MTLHeap`s**
(verified on-device and in `MPSAllocator.mm` — pools built with
`UsageFlags::HAZARD`, bound by torch itself via `setBuffer:` so torch never
needs `useResource:`). macOS evicts idle GPU memory after roughly a second
(independently documented in ggml-org/llama.cpp#10119, fixed there with
`MTLResidencySet`, llama.cpp PR #11427), after which the undeclared heap's
`gpuAddress` silently reads as zeros and writes to it are dropped.

## The red herring we chased first

The bug originally presented as: *"the first kernel dispatch after
`mojo --clear-cache -f` returns all-zero output, then self-heals."* That
framing is wrong — the cache is only a **delay amplifier**:

- cold-cache `mojo build`: several seconds between torch's last GPU work
  and the dispatch → heap evicted → **always fails**;
- warm-cache build: ~1.2 s → borderline inside the window → passes;
- second call in-process: ~0 s gap → passes ("self-healing");
- **any** ≥1.5 s GPU-idle gap reproduces it with a fully warm cache and a
  prebuilt `.so` — e.g. `time.sleep(2)` before the dispatch, or think-time
  between decode steps in an interactive workload. Sub-second gaps never do.

We bisected the threshold on this machine to between 1.0 s (passes) and
1.5 s (fails), matching the llama.cpp report of macOS's ~1 s idle eviction.

## Environment

- Mac mini (Mac16,10), Apple M4, 16 GB, macOS 26.3.2 (25D2140)
- Mojo `1.0.0b3.dev2026062306` (`b65c36ac`), Metal backend
- PyTorch 2.8.0 (only for `repro.py`/`workaround.py`; the `.mojo` repros
  have zero Python/PyTorch dependency)

## Repros (all in this branch, `mojo-cold-cache-bug-repro`)

1. **`repro_heap.mojo` — pure Mojo, deterministic, matches the real-world
   case.** Allocates a hazard-tracked shared `MTLHeap` via Obj-C FFI
   (exactly PyTorch's configuration), sub-allocates two tiny buffers,
   touches the heap with a blit on a separate queue (playing PyTorch's
   role), sleeps 2 s, then dispatches a one-thread copy kernel against the
   raw `gpuAddress`es. `uv run mojo run repro_heap.mojo` → the sentinel in
   the output buffer is untouched: the dispatch did nothing. Toggles in
   the file demonstrate the full behavior matrix (no-touch → fails even
   without sleeping; touch + no sleep → passes).
2. **`repro_pure.mojo` — pure Mojo, deterministic, sibling case.** Same
   dispatch against a *device-allocated* buffer with
   `MTLResourceHazardTrackingModeUntracked` (no heap, no sleep needed):
   always fails; flip one bit to hazard-tracked and it always passes.
3. **`repro.py` — the original PyTorch-interop manifestation.**
   `mojo --clear-cache -f && uv run python repro.py` → all-zero output.
   The git history of this branch documents the step-by-step trimming from
   a real `causal_conv1d_update` kernel down to this ~25-line script, each
   step verified to still reproduce.
4. **`workaround.py`** — same flow as `repro.py` plus the mitigation below;
   passes even on a cold cache with an extra 2 s sleep.

## Evidence

1. **The dispatch is truly lost, not a stale view**: prefilling the output
   tensor with a sentinel (99) shows it untouched after the "failed"
   dispatch, read back both through torch and directly through the
   `MTLBuffer`'s `contents` pointer.
2. **Per-heap independence / third failure mode**: with the input tensor on
   an idle large-pool heap and the output tensor on a freshly-revived
   small-pool heap, the kernel *runs* — the write lands, but the read
   returns zeros (output becomes 0.0 instead of staying 99). Reads and
   writes fail independently per undeclared heap.
3. **Ground truth on PyTorch buffers** (Obj-C introspection on-device):
   every MPS tensor's `MTLBuffer` reports `hazardTrackingMode=2 (Tracked)`,
   `storageMode=0 (Shared)`, `[buffer heap] != nil`
   (`AGXG16GFamilyHeap`, 8 MiB small pool / 32 MiB large pool), on the
   same `MTLDevice` object as `MTLCreateSystemDefaultDevice()`.
4. **Disassembly of `libAsyncRTMojoBindings.dylib`** (the only dylib in the
   Mojo distribution linking Metal.framework; contains the source path
   string `MLRT/lib/Driver/DeviceContext/Metal/MetalDeviceContext.cpp`):
   of the 221 real `objc_msgSend` call sites in the binary, 203 were
   resolved to concrete selectors. Exactly **two** call
   `useResource:usage:`, both inside `enqueueFunctionExecDirect`, and both
   are gated behind a lookup of the argument's GPU address in a
   mutex-protected, address-range-indexed internal allocation table —
   entries exist only for Mojo-created buffers, and the matched entry's
   `id<MTLBuffer>` is what gets declared. No call sites exist for
   `useHeap:`, `useResidencySet:`, `addResidencySet:` or any
   `MTLResidencySet` machinery (those selectors appear only in a generated
   catch-all selector catalog). Kernel arguments are bound with
   `setBuffer:offset:atIndex:` in a loop; the dispatch is the classic
   `dispatchThreadgroups:threadsPerThreadgroup:` + `commit` +
   `waitUntilCompleted` sequence with no fences/events.
5. **Silent failure**: `MTL_DEBUG_LAYER=1` (API validation) reports nothing
   for the lost dispatch. There is no command-buffer error. (Aside:
   `MTL_SHADER_VALIDATION=1` breaks Mojo's runtime pipeline creation
   entirely — `Failed to create compute pipeline state ... XPC_ERROR_
   CONNECTION_INTERRUPTED` — which is its own minor issue and prevented us
   from using shader validation to observe the bad access directly.)

## Workaround (validated, in `workaround.py`)

Immediately before every Mojo dispatch, issue a tiny GPU op touching
**each** argument tensor, then synchronize:

```python
for t in (x, out):            # every tensor argument, not just one:
    t.view(-1)[0:1].add_(0)   # heaps are revived individually
torch.mps.synchronize()
variant_fn(gpu_address(x), gpu_address(out))
```

This puts the dispatch back inside the driver's residency window. Costs a
couple of tiny kernel launches (~tens of µs). A bare
`torch.mps.synchronize()` is NOT sufficient (an empty queue submits
nothing); the touch must be real GPU work. For tensors requiring grad, use
a read-only touch (`t.view(-1)[0:1].clone()`) to avoid the autograd
version-counter bump.

## Suggested fixes on the Mojo side

Any of, in increasing order of niceness:

1. Document loudly that `unsafe_from_address` pointers into GPU memory not
   allocated by the `DeviceContext` are not supported on Metal (they
   *appear* to work, which is the trap).
2. Provide an API to import/register an external `MTLBuffer` (or address
   range) with the `DeviceContext`, inserting it into the existing
   allocation-tracking table so the already-present `useResource:` path
   covers it.
3. Attach an `MTLResidencySet` to the command queue covering all argument
   address ranges the runtime cannot resolve — or simply follow llama.cpp
   (PR #11427) in using residency sets wholesale.

## Reproduction quickstart

```bash
git checkout mojo-cold-cache-bug-repro && uv sync

uv run mojo run repro_heap.mojo        # pure Mojo, deterministic  -> FAIL
uv run mojo run repro_pure.mojo        # pure Mojo, untracked case -> FAIL
uv run mojo --clear-cache -f
uv run python repro.py                 # PyTorch manifestation     -> FAIL
uv run python workaround.py            # mitigation                -> PASS
```
