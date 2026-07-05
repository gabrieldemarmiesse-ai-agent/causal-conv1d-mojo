"""Shared `_silu_f32` helper for the GPU and CPU subpackages.

Lives at the package root so every subpackage (`fwd`, `update`,
`fwd_cpu`, `update_cpu`) can pull it in via the `_PKG_DIR` entry in
the `include_dirs=...` passed to `_jit_common.compile_and_load`.
"""

from std.math import exp, recip


def _silu_f32[w: Int = 1](
    x: SIMD[DType.float32, w]
) -> SIMD[DType.float32, w]:
    """SiLU activation: `x * sigmoid(x)`, elementwise over `w` lanes.

    Expressed as `x * recip(1 + exp(-x))` so the division lowers to
    a single fast reciprocal (`rcp.approx.ftz.f32` on nvptx, `v_rcp_f32`
    on amdgcn) + multiply rather than the full IEEE-compliant divide
    expansion (~12 instructions on amdgcn). The ~1 ulp accuracy loss
    on the reciprocal is well within the silu tolerance — all GPU
    tests pass with the same numerical bounds.

    Scalar callers keep the old `_silu_f32(x)` spelling (`w` defaults
    to 1 and `Float32` is `SIMD[f32, 1]`); the vectorized CPU kernels
    bind the width explicitly (`_silu_f32[kV](acc)`).
    """
    return x * recip(SIMD[DType.float32, w](1) + exp(-x))
