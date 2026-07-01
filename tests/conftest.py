import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_rng():
    torch.manual_seed(0)
