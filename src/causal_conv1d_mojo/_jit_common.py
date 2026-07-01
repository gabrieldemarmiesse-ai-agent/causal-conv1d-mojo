"""Shared JIT-on-first-use infrastructure (trimmed repro; Metal only)."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import platform
import sys
from pathlib import Path
from types import ModuleType
from collections.abc import Iterable, Mapping

from mojo.run import subprocess_run_mojo

_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"))


def cache_dir_for(subpkg: str, backend: str, backend_arch: str = "") -> Path:
    return _CACHE_HOME / "causal_conv1d_mojo" / subpkg / backend / backend_arch


def compile_and_load(
    *,
    subpkg: str,
    source_file: Path,
    include_dirs: Iterable[Path] = (),
    defines: Mapping[str, str] = {},
    mod_name: str,
    backend: str,
    backend_arch: str = "",
) -> ModuleType:
    include_dirs = [Path(d) for d in include_dirs]
    cache_dir = cache_dir_for(subpkg, backend, backend_arch)
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_hash = _hash_sources(source_file, include_dirs, defines)
    so_path = cache_dir / f"{mod_name}.hash-{src_hash}.so"

    if not so_path.is_file():
        for old in cache_dir.glob(f"{mod_name}.hash-*.so"):
            old.unlink()
        print(
            f"[causal_conv1d_mojo] compiling {subpkg} variant {mod_name} — "
            f"cached for future runs.",
            file=sys.stderr,
            end="",
        )
        cmd = ["build", str(source_file)]
        for d in include_dirs:
            cmd += ["-I", str(d)]
        for k, v in defines.items():
            cmd += ["-D", f"{k}={v}"]
        cmd += ["--emit", "shared-lib", "-o", str(so_path)]
        subprocess_run_mojo(cmd, check=True)
        print(" done", file=sys.stderr)

    loader = importlib.machinery.ExtensionFileLoader("variant", str(so_path))
    spec = importlib.util.spec_from_loader("variant", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[mod_name] = module
    return module


def detect_gpu_backend() -> tuple[str, str]:
    macos_major = (platform.mac_ver()[0] or "").split(".")[0]
    return ("metal", f"macos{macos_major}")


def _hash_sources(
    source_file: Path,
    include_dirs: Iterable[Path],
    defines: Mapping[str, str],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(source_file.read_bytes())
    for d in include_dirs:
        for f in sorted(Path(d).glob("*.mojo")):
            hasher.update(f.read_bytes())
    for k in sorted(defines):
        hasher.update(f"{k}={defines[k]}\n".encode())
    return hasher.hexdigest()[:16]
