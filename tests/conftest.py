import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)


@pytest.fixture(params=[4], ids=["w4"])
def width(request):
    return request.param
