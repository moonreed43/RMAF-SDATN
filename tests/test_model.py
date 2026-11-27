import numpy as np
import pytest
import torch

from rmaf_sdatn.models.forecaster import Forecaster
from rmaf_sdatn.models.lora import LoRA


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))])
def test_adapter_gradients_shapes_and_merged_prediction(config, device):
    model = Forecaster(config["model"]).to(device).requires_grad_(False)
    adapter = LoRA(model.backbone, 4, 8).to(device)
    assert set(adapter.matrices) == {"weight_ih_l0", "weight_hh_l0", "weight_ih_l1", "weight_hh_l1"}
    for key, residual in adapter.residuals().items():
        assert residual.shape == getattr(model.backbone, key).shape
    x = torch.rand(3, 12, 1, device=device)
    first = model(x, adapter.residuals()).power
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-2)
    first.mean().backward()
    assert all(param.grad is None for param in model.parameters())
    assert all(m.b.grad.abs().sum() > 0 for m in adapter.matrices.values())
    optimizer.step()
    model.adapter = adapter
    model.eval()
    prediction = model(x)
    assert prediction.power.shape == (3, 1)
    assert prediction.shared.shape == (3, 6) and prediction.supplementary.shape == (3, 5)
    assert (prediction.power > 0).all()
    torch.testing.assert_close(prediction.attention.sum(1), torch.ones(3, device=device))
    for i, scale in enumerate(config["model"]["scales"]):
        assert prediction.magnitudes[:, :scale, i].count_nonzero() == 0
    assert not torch.allclose(first, prediction.power)
    torch.testing.assert_close(prediction.power, model.merged()(x).power, rtol=1e-5, atol=1e-6)


def test_concatenated_rank_matches_weighted_effective_matrices(config):
    model = Forecaster(config["model"])
    adapters = [LoRA(model.backbone, 2, 8), LoRA(model.backbone, 3, 8)]
    for adapter in adapters:
        for matrix in adapter.matrices.values():
            torch.nn.init.normal_(matrix.b)
    weights = np.array([.3, .7])
    fused = LoRA.fuse(model.backbone, adapters, weights)
    assert fused.rank == 5
    for name, result in fused.residuals().items():
        expected = .3*adapters[0].matrices[name]() + .7*adapters[1].matrices[name]()
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
