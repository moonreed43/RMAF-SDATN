import pytest
import numpy as np
import random
import torch
from torch import nn

from rmaf_sdatn.evaluation.modern_baselines import modern_baseline
from rmaf_sdatn.training.common import Training, configure


def test_epoch_resume_restores_optimizer_and_sampling_state(config, tmp_path):
    training = config["training"]

    def run(directory, interrupt=False):
        tr = Training(config, 7, configure(7, training), directory)
        model = nn.Linear(1, 1)
        optimizer = tr.adam(model.parameters())

        def epoch(i):
            if interrupt and i == 1:
                raise InterruptedError
            x = torch.rand(8, 1, generator=tr.generator)
            y = torch.rand(8, 1) * (random.random() + float(np.random.uniform()))
            return {"loss": float(tr.step((model(x)-y).square().mean(), optimizer))}

        tr.early_fit(3, [model], epoch, lambda: float(model.weight.detach().square().sum()), optimizer, "check")
        return model.state_dict()

    expected = run(tmp_path/"uninterrupted")
    with pytest.raises(InterruptedError):
        run(tmp_path/"resumed", interrupt=True)
    resumed = run(tmp_path/"resumed")
    for param, tensor in expected.items():
        torch.testing.assert_close(tensor, resumed[param], rtol=0, atol=0)


@pytest.mark.parametrize("method", ["patchtst", "timesnet"])
def test_modern_backbone_direct_multioutput(config, method):
    config["model"]["horizon"] = 24
    model = modern_baseline(config, method)
    prediction = model(torch.rand(4, 12, 1)).power
    assert prediction.shape == (4, 24)
    torch.nn.functional.mse_loss(prediction, torch.rand_like(prediction)).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
