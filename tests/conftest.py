import pytest
import torch

from rmaf_sdatn.config import load_config


@pytest.fixture
def config():
    torch.set_num_threads(1)
    cfg = load_config("configs/experiments/main.yaml")
    cfg["model"].update(hidden=8, shared_dim=6, supplementary_dim=5, embedding=4, discriminator_hidden=8)
    cfg["training"]["device"] = "cpu"
    return cfg
