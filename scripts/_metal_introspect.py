#!/usr/bin/env python3
"""Headless Metal kernel introspection — static perf-relevant facts about
a Mojo-compiled Metal kernel with no Instruments GUI.

Recovers, all without the Instruments GUI (confirmed on Apple M4 / macOS 26):
  - threadgroup (shared) memory bytes, SIMD width, max threads/threadgroup
    (MTLComputePipelineState reflection, via ctypes → Metal)
  - AIR instruction-mix histogram (`metal-objdump -d air64` on the
    metallib Mojo caches under ~/.cache/modular) — the Apple analog of the
    CUDA PTX histogram the master bench diffs on NVIDIA
  - native AGX (Apple GPU) machine-code extraction, best-effort
    (MTLBinaryArchive → metal-source), reported as a code-size proxy

NOT reachable headlessly on this hardware (empirically settled, kept here
so nobody re-chases it):
  - hardware occupancy / ALU% / stall counters: device.counterSets exposes
    only 'timestamp'; the rich sets are Instruments-GUI-only, and kperf is
    CPU-PMU-only.
  - register-count→occupancy inference: Apple9/M4 "Dynamic Caching"
    allocates registers per-PC rather than at a fixed peak, so the M1-era
    maxTotalThreadsPerThreadgroup stairstep does not hold; the community
    `applegpu` disassembler has no G16/M4 support; and the compiler's
    __GPU_STATS_MD flatbuffer has no public schema.

Depends only on the stdlib + ctypes + the Metal toolchain (`xcrun
metal-objdump` / `metal-source` / `metal-lipo`) and Metal.framework — no
torch, no venv packages — so it runs under a plain `python3`.

    python3 scripts/_metal_introspect.py --fn fwd        # find + report
    python3 scripts/_metal_introspect.py path/to.metallib
    python3 scripts/_metal_introspect.py --fn fwd --json  # for tooling
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Kernel-function-name substrings per benched function (the metallib symbol
# is `kernel_<fn>_...`). Mirrors master_bench.SUBPKG's intent.
_FN_SYMBOL = {"fwd": "fwd_kernel", "bwd": "bwd_full_kernel", "update": "update_kernel"}

_MOJO_CACHE = Path.home() / ".cache" / "modular" / ".mojo_cache"


# --------------------------------------------------------------------------
# Metallib discovery
# --------------------------------------------------------------------------
def _is_metallib(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"MTLB"
    except OSError:
        return False


def _function_name(metallib: str) -> str | None:
    """First (only) function symbol in the metallib, via metal-nm."""
    r = subprocess.run(["xcrun", "metal-nm", metallib], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "T":
            return parts[2]
    return None


def find_kernel_metallib(fn: str | None) -> str | None:
    """Newest cached metallib whose function symbol matches `fn` (fwd/bwd/
    update), or newest overall if `fn` is None. Scans ~/.cache/modular."""
    want = _FN_SYMBOL.get(fn) if fn else None
    candidates = sorted(
        (p for p in _MOJO_CACHE.rglob("*") if p.is_file() and _is_metallib(p)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        if want is None:
            return str(p)
        sym = _function_name(str(p))
        if sym and want in sym:
            return str(p)
    return None


# --------------------------------------------------------------------------
# ctypes → Objective-C / Metal bridge (no torch, no venv)
# --------------------------------------------------------------------------
_objc = None


def _init_objc():
    global _objc
    if _objc is not None:
        return _objc
    lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    lib.sel_registerName.restype = ctypes.c_void_p
    lib.sel_registerName.argtypes = [ctypes.c_char_p]
    lib.objc_getClass.restype = ctypes.c_void_p
    lib.objc_getClass.argtypes = [ctypes.c_char_p]
    _objc = lib
    return lib


def _sel(n):
    return _init_objc().sel_registerName(n.encode())


def _msg(recv, name, *args, restype=ctypes.c_void_p, argtypes=None):
    fn = _init_objc().objc_msgSend
    fn.restype = restype
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + (argtypes or [])
    return fn(recv, _sel(name), *args)


def _nsstr(s):
    return _msg(
        _init_objc().objc_getClass(b"NSString"),
        "stringWithUTF8String:",
        s.encode(),
        argtypes=[ctypes.c_char_p],
    )


def _to_str(p):
    if not p:
        return None
    obj = _init_objc()
    obj.objc_msgSend.restype = ctypes.c_char_p
    obj.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    r = obj.objc_msgSend(p, _sel("UTF8String"))
    obj.objc_msgSend.restype = ctypes.c_void_p
    return r.decode() if r else None


def _uint(recv, name):
    return _msg(recv, name, restype=ctypes.c_ulong)


def _url(path):
    return _msg(
        _init_objc().objc_getClass(b"NSURL"),
        "fileURLWithPath:",
        _nsstr(path),
        argtypes=[ctypes.c_void_p],
    )


def pipeline_static_facts(metallib: str) -> dict:
    """MTLComputePipelineState reflection for the metallib's kernel."""
    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    dev = metal.MTLCreateSystemDefaultDevice()
    cs = _msg(dev, "counterSets")
    counter_sets = [
        _to_str(_msg(_msg(cs, "objectAtIndex:", i, argtypes=[ctypes.c_ulong]), "name"))
        for i in range(_uint(cs, "count") if cs else 0)
    ]
    out = {
        "device": _to_str(_msg(dev, "name")),
        "counter_sets": counter_sets,
    }
    err = ctypes.c_void_p(0)
    lib = _msg(
        dev,
        "newLibraryWithURL:error:",
        _url(metallib),
        ctypes.byref(err),
        argtypes=[ctypes.c_void_p, ctypes.c_void_p],
    )
    if not lib:
        out["error"] = _to_str(_msg(err, "localizedDescription"))
        return out
    names = _msg(lib, "functionNames")
    if not _uint(names, "count"):
        out["error"] = "no functions in metallib"
        return out
    fn = _to_str(_msg(names, "objectAtIndex:", 0, argtypes=[ctypes.c_ulong]))
    func = _msg(lib, "newFunctionWithName:", _nsstr(fn), argtypes=[ctypes.c_void_p])
    err = ctypes.c_void_p(0)
    pso = _msg(
        dev,
        "newComputePipelineStateWithFunction:error:",
        func,
        ctypes.byref(err),
        argtypes=[ctypes.c_void_p, ctypes.c_void_p],
    )
    if not pso:
        out["error"] = _to_str(_msg(err, "localizedDescription"))
        return out
    out.update(
        function=fn,
        threadgroup_memory_bytes=_uint(pso, "staticThreadgroupMemoryLength"),
        simd_width=_uint(pso, "threadExecutionWidth"),
        max_threads_per_threadgroup=_uint(pso, "maxTotalThreadsPerThreadgroup"),
    )
    return out


# --------------------------------------------------------------------------
# AIR instruction-mix histogram
# --------------------------------------------------------------------------
def air_instruction_histogram(metallib: str) -> dict[str, int]:
    """Opcode histogram of the metallib's AIR (LLVM-IR-level, pre-register-
    allocation — the Apple analog of a PTX histogram)."""
    txt = subprocess.run(
        ["xcrun", "metal-objdump", "-d", "--arch-name=air64", metallib],
        capture_output=True,
        text=True,
    ).stdout
    ops: Counter = Counter()
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith(("br ", "store ", "ret ", "switch ", "unreachable")):
            ops[s.split(" ", 1)[0].rstrip(",")] += 1
        elif "= " in s:
            rhs = s.split("= ", 1)[1].lstrip()
            op = rhs.split(" ", 1)[0].split("(", 1)[0]
            if op[:1].isalpha() and op.replace("_", "").isalnum():
                ops[op] += 1
    return dict(ops)


# --------------------------------------------------------------------------
# Native AGX extraction (best-effort code-size proxy)
# --------------------------------------------------------------------------
def extract_native_agx(metallib: str, workdir: str = "/tmp/ccv_agx") -> dict:
    """MTLBinaryArchive → serialized fat file → thin the applegpu slice.
    Returns {slice, text_bytes} or {error}. metal-objdump has no g16
    instruction printer, so only the code SIZE is reported, not a
    disassembly."""
    metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    dev = metal.MTLCreateSystemDefaultDevice()
    err = ctypes.c_void_p(0)
    lib = _msg(
        dev,
        "newLibraryWithURL:error:",
        _url(metallib),
        ctypes.byref(err),
        argtypes=[ctypes.c_void_p, ctypes.c_void_p],
    )
    if not lib:
        return {"error": "library load failed"}
    names = _msg(lib, "functionNames")
    fn = _to_str(_msg(names, "objectAtIndex:", 0, argtypes=[ctypes.c_ulong]))
    func = _msg(lib, "newFunctionWithName:", _nsstr(fn), argtypes=[ctypes.c_void_p])
    desc = _msg(
        _msg(_init_objc().objc_getClass(b"MTLBinaryArchiveDescriptor"), "alloc"), "init"
    )
    err = ctypes.c_void_p(0)
    archive = _msg(
        dev,
        "newBinaryArchiveWithDescriptor:error:",
        desc,
        ctypes.byref(err),
        argtypes=[ctypes.c_void_p, ctypes.c_void_p],
    )
    cpd = _msg(
        _msg(_init_objc().objc_getClass(b"MTLComputePipelineDescriptor"), "alloc"),
        "init",
    )
    _msg(cpd, "setComputeFunction:", func, argtypes=[ctypes.c_void_p])
    err = ctypes.c_void_p(0)
    if not _msg(
        archive,
        "addComputePipelineFunctionsWithDescriptor:error:",
        cpd,
        ctypes.byref(err),
        restype=ctypes.c_bool,
        argtypes=[ctypes.c_void_p, ctypes.c_void_p],
    ):
        return {"error": "addComputePipelineFunctions failed"}
    fat = f"{workdir}.archive"
    err = ctypes.c_void_p(0)
    if not _msg(
        archive,
        "serializeToURL:error:",
        _url(fat),
        ctypes.byref(err),
        restype=ctypes.c_bool,
        argtypes=[ctypes.c_void_p, ctypes.c_void_p],
    ):
        return {"error": "serializeToURL failed"}
    info = subprocess.run(
        ["xcrun", "metal-lipo", "-info", fat], capture_output=True, text=True
    ).stdout
    agx = next(
        (s for s in info.split("are:")[-1].split() if s.startswith("applegpu")), None
    )
    if not agx:
        return {"error": "no applegpu slice"}
    thin = f"{workdir}.{agx}"
    subprocess.run(
        ["xcrun", "metal-lipo", fat, "-thin", agx, "-output", thin], check=False
    )
    sec = subprocess.run(
        ["xcrun", "metal-objdump", "--section-headers", thin],
        capture_output=True,
        text=True,
    ).stdout
    # The lipo-thinned applegpu slice exposes the native code as a
    # __compute section (a nested Mach-O whose own __text is the real
    # instruction stream). Sum the __compute sections as the code-size
    # proxy — Apple ships no g16 instruction printer, so size is all we get.
    code_bytes = 0
    for line in sec.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "__compute":
            try:
                code_bytes += int(parts[2], 16)
            except ValueError:
                pass
    return {"slice": agx, "code_bytes": code_bytes, "path": thin}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def introspect(metallib: str) -> dict:
    rep: dict = {"metallib": metallib}
    try:
        rep["static"] = pipeline_static_facts(metallib)
    except Exception as e:  # noqa: BLE001
        rep["static"] = {"error": str(e)}
    try:
        rep["air_histogram"] = air_instruction_histogram(metallib)
    except Exception as e:  # noqa: BLE001
        rep["air_histogram"] = {"error": str(e)}
    try:
        rep["agx"] = extract_native_agx(metallib)
    except Exception as e:  # noqa: BLE001
        rep["agx"] = {"error": str(e)}
    return rep


def _print_human(rep: dict) -> None:
    print(f"metallib: {rep['metallib']}")
    s = rep.get("static", {})
    print(f"  device: {s.get('device')}   counter sets: {s.get('counter_sets')}")
    if "function" in s:
        print(f"  function: {s['function']}")
        print(f"    threadgroup memory : {s['threadgroup_memory_bytes']} bytes")
        print(f"    SIMD width         : {s['simd_width']}")
        print(
            f"    max threads/tg     : {s['max_threads_per_threadgroup']}  "
            "(note: reflects Mojo's baked MAX_THREADS attribute, not a raw "
            "register ceiling; occupancy is not statically recoverable on M4)"
        )
    elif s.get("error"):
        print(f"    (static facts unavailable: {s['error']})")
    hist = rep.get("air_histogram", {})
    if hist and "error" not in hist:
        print("  AIR instruction mix (pre-regalloc; analog of PTX):")
        for op, c in sorted(hist.items(), key=lambda kv: -kv[1])[:16]:
            print(f"    {c:4d}  {op}")
    agx = rep.get("agx", {})
    if agx.get("slice"):
        print(
            f"  native AGX slice: {agx['slice']}  "
            f"({agx['code_bytes']} bytes of machine code; no g16 disassembler)"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("metallib", nargs="?", help="path to a .metallib")
    ap.add_argument(
        "--fn",
        choices=("fwd", "bwd", "update"),
        help="find newest cached kernel for this fn",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    mll = args.metallib or find_kernel_metallib(args.fn)
    if not mll:
        print(
            "no metallib found (run the kernel first, or pass a path)", file=sys.stderr
        )
        return 1
    rep = introspect(mll)
    if args.json:
        print(json.dumps(rep))
    else:
        _print_human(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
