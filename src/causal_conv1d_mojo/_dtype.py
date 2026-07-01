"""Shared helpers used by every per-function subpackage wrapper."""

from __future__ import annotations

import torch

# Must match the dispatch in the Mojo entry points.
_DTYPE_CODE = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
}
