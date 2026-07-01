"""Compile-and-load helper (trimmed repro: no on-disk caching — always
rebuild fresh, since this repro only ever calls the kernel once)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from collections.abc import Iterable

from mojo.run import subprocess_run_mojo


def compile_and_load(
    *,
    source_file: Path,
    include_dirs: Iterable[Path] = (),
) -> ModuleType:
    so_path = Path(tempfile.mkstemp(suffix=".so")[1])
    cmd = ["build", str(source_file)]
    for d in include_dirs:
        cmd += ["-I", str(d)]
    cmd += ["--emit", "shared-lib", "-o", str(so_path)]
    print("[causal_conv1d_mojo] compiling update kernel...", file=sys.stderr, end="")
    subprocess_run_mojo(cmd, check=True)
    print(" done", file=sys.stderr)

    loader = importlib.machinery.ExtensionFileLoader("variant", str(so_path))
    spec = importlib.util.spec_from_loader("variant", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module
