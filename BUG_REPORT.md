# Mojo GPU kernel reads/writes zero on Apple Metal for buffers it didn't allocate

## Summary

A Mojo kernel launched via `DeviceContext.enqueue_function` against a
GPU buffer it did not allocate itself (reconstructed from a raw address
via `UnsafePointer(unsafe_from_address=...)`) can silently read/write
zero instead of the real data, on Apple Metal.

We have two repros:

1. **`repro.py`** — the real-world case: a PyTorch MPS tensor's GPU
   address is extracted (via the same `gpuAddress` Objective-C selector
   Mojo itself uses internally) and handed to a Mojo kernel. This fails
   deterministically on the *first* kernel dispatch after a cold
   `~/.cache/modular` cache, then **self-heals** — a second call in the
   same process, or any call once the cache is warm, succeeds.
2. **`repro_pure.mojo`** — a from-scratch, dependency-free isolation: a
   raw `MTLBuffer` allocated directly via Objective-C
   (`newBufferWithLength:options:`) with
   `MTLResourceHazardTrackingModeUntracked` set, then handed to a Mojo
   kernel the same way. This fails **100% of the time**, independent of
   cache state or repeated calls — flip the one `options` bit back to
   the default (hazard-tracked) and it always succeeds instead.

We believe (1) and (2) are related but have not been able to fully
unify them into one root cause — see "Open question" below.

## Environment

- Hardware: Mac mini (Mac16,10), Apple M4, 16 GB
- OS: macOS 26.3.2 (build 25D2140)
- Mojo: `1.0.0b3.dev2026062306` (`b65c36ac`)
- PyTorch: `2.8.0` (only used by `repro.py`; `repro_pure.mojo` has zero
  Python/PyTorch dependency)
- Backend: Metal (`mps`)

## Repro 1: PyTorch MPS interop (`repro.py`), flaky / self-healing

```bash
git clone <this repo>
cd causal-conv1d-mojo
git checkout mojo-cold-cache-bug-repro
uv sync
uv run mojo --clear-cache -f     # clear Mojo's own toolchain cache
uv run python repro.py
```

Expected: prints a nonzero output. Actual: prints `tensor([0.])` and the
script's own assertion fails, on a cold cache. Run it again (or run it
twice in the same process — see `/tmp`-style variant below) and it
passes.

The whole call chain (`src/causal_conv1d_mojo/`) is trimmed to ~150
lines total across `__init__.py`, `kernel.mojo`, and `variant.mojo` —
see the file headers for how the GPU address is extracted from a torch
tensor and handed to the kernel. The kernel itself is a single-thread,
single-element copy (`o_ptr[0] = x_ptr[0]`); the bug reproduces
regardless of what the kernel actually computes.

Every commit on this branch is a verified-still-reproducing trim step,
if you want to see the full derivation from the original, much larger
`causal_conv1d_update` kernel down to this minimal form.

## Repro 2: pure Mojo, deterministic (`repro_pure.mojo`)

```bash
uv run mojo run repro_pure.mojo
```

No PyTorch, no Python, no `mojo build`/cache interaction at all beyond
what `mojo run` itself does. The program:

1. Calls `MTLCreateSystemDefaultDevice()` and `newBufferWithLength:2
   options:256` directly via `objc_msgSend` (256 =
   `MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked`)
   to allocate two 1-element fp16 buffers — entirely outside of Mojo's
   own `DeviceContext.enqueue_create_buffer` allocator.
2. Writes a known value (42.0) into the input buffer via its Metal
   `contents` pointer.
3. Extracts each buffer's `gpuAddress` and wraps it in an
   `UnsafePointer(unsafe_from_address=...)`.
4. Creates a Mojo `DeviceContext()`, compiles a trivial one-thread copy
   kernel, and dispatches it against those two pointers.
5. Reads the output buffer back via its `contents` pointer.

Result: **always** prints `out: 0.0` — the copy silently did nothing.
Change `kUntrackedShared` from `256` to `0` (dropping
`MTLResourceHazardTrackingModeUntracked`) and the same kernel, same
dispatch code, same buffers-not-owned-by-DeviceContext setup **always**
prints `out: 42.0` instead. We ran each configuration 3+ times, with
and without clearing `~/.cache/modular`, and across fresh processes and
repeated in-process dispatches — the tracked/untracked bit is the only
variable that changes the outcome, 100% of the time in both directions.

## Analysis

`MTLResourceHazardTrackingModeUntracked` tells Metal the *application*
is responsible for any synchronization/visibility guarantees around
that buffer — normally via `[MTLComputeCommandEncoder useResource:
usage:]` (or a residency set) before encoding a dispatch that touches
it. ML frameworks commonly allocate their tensor storage with this flag
for performance (we have not directly confirmed PyTorch's MPS allocator
does this, but it is standard practice and consistent with what we
observe).

Our reading is that `DeviceContext.enqueue_function` does not call
`useResource:` (or add the buffer to a residency set) for pointer
arguments that were never obtained through Mojo's own buffer allocator
(i.e. arguments constructed via `unsafe_from_address` rather than a
`DeviceBuffer` handle). For a hazard-tracked buffer this is harmless —
Metal's automatic tracking covers it regardless of how Mojo refers to
it. For a hazard-*un*tracked buffer, nothing declares the dependency to
the GPU, and the dispatch can execute without the correct
read/write visibility, observed here as the kernel appearing to do
nothing (reads/writes zero).

## Open question: does this explain the flaky PyTorch case?

We could not fully reconcile the two repros' behavior:

- Repro 2 (pure Mojo, untracked hazard buffer) fails **every time**,
  regardless of cache state or how many times the same kernel is
  dispatched in the same process.
- Repro 1 (PyTorch interop) fails only on the **first** dispatch after
  a cold `~/.cache/modular`, then **self-heals** on any subsequent
  call — even though each call in `repro.py` creates a brand-new
  `DeviceContext()` and a freshly-recompiled `.so` (we deliberately
  removed all caching/reuse of the Mojo context and compiled artifact
  across calls while trimming this repro, and the self-healing
  behavior persisted regardless).

We verified the self-healing behavior holds even with that caching
removed, which rules out "Mojo reuses the same compiled kernel/context
object" as the explanation. Our best guess is that Mojo's own toolchain
cache (`~/.cache/modular/.mojo_cache`) being warm changes how fast/how
`mojo build` re-compiles the *same* kernel source, and that in turn
affects something downstream at the Metal driver level (e.g. Apple's
own system-level shader compilation cache, or GPU clock/wake state) —
a timing-sensitive interaction layered on top of the same
untracked-hazard-buffer gap identified in Repro 2. We were not able to
fully verify this within pure Mojo, since Repro 2 does not depend on
`mojo build`/caching at all (it's a single `mojo run` invocation).

We think Repro 2 stands on its own as a clear, deterministic bug
(`DeviceContext.enqueue_function` does not establish Metal residency
for hazard-untracked foreign buffers), and Repro 1 is very likely the
same underlying gap manifesting through PyTorch's real allocator, with
an additional cache/timing factor we have not fully isolated.

## Files

- `repro.py` — PyTorch MPS interop repro (flaky/self-healing)
- `repro_pure.mojo` — pure-Mojo deterministic repro
- `src/causal_conv1d_mojo/` — the trimmed-down Python/Mojo glue used by
  `repro.py` (`__init__.py`, `kernel.mojo`, `variant.mojo`)
- Git history on this branch (`mojo-cold-cache-bug-repro`) documents
  every trimming step from the original, much larger
  `causal_conv1d_update` kernel down to this minimal form, each
  verified to still reproduce the bug before moving to the next.
