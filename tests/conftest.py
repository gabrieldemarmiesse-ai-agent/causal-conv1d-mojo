from beartype.claw import beartype_package  # noqa: I001

beartype_package("causal_conv1d_mojo")

import pytest  # noqa: E402
import torch  # noqa: E402

import causal_conv1d_mojo._update as _update_mod  # noqa: E402

_update_mod._MPS_UPDATE_FALLBACK_THRESHOLD = 0


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


@pytest.fixture(params=["mps"])
def device(request):
    return request.param


@pytest.fixture(params=["silu"])
def activation(request):
    return request.param


@pytest.fixture(params=[True], ids=["with_bias"])
def bias_present(request):
    return request.param


@pytest.fixture(params=[torch.float16], ids=["fp16"])
def dtype(request):
    return request.param


@pytest.fixture(params=[4], ids=["w4"])
def width(request):
    return request.param
