import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


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
