import numpy as np
import torch

from rmaf_sdatn.algorithms.alignment import ReverseGradient, domain_loss, sample_gates
from rmaf_sdatn.algorithms.anchoring import anchor_loss, snapshot, strengths
from rmaf_sdatn.algorithms.fusion import kl_prior, project_simplex, search
from rmaf_sdatn.algorithms.reliability import discrepancy, reliability
from rmaf_sdatn.models.forecaster import DomainDiscriminator, Forecaster


def test_discrepancy_and_positive_reliability():
    target = torch.tensor([[0.], [2.]])
    source = torch.tensor([[1.], [5.]])
    assert float(discrepancy(source, target, 1)) == 40
    vals = torch.tensor([0., 0.5, 1.])
    rho, delta, error = reliability(vals, vals, .5, .5, 1e-8)
    torch.testing.assert_close(rho, torch.softmax(-2*vals.double(), dim=0))
    torch.testing.assert_close(reliability(torch.ones(3), torch.ones(3), .5, .001, 1e-8)[0],
                               torch.full((3,), 1/3, dtype=torch.float64))
    assert rho.min() > 0 and rho.argmax() == 0 and float(rho.sum()) == 1


def test_search_evaluates_projected_population_and_keeps_best(config):
    np.testing.assert_allclose(project_simplex([2, 0.1, -2]), [1, 0, 0])
    target = np.array([.45, .55])
    calls = []

    def objective(value):
        assert np.min(value) >= 0
        np.testing.assert_allclose(value.sum(), 1)
        calls.append(value.copy())
        return np.square(value-target).sum()

    prior = np.array([.9, .1])
    settings = dict(config["algorithm"]["fusion"], population=8, generations=12, mutation_std=.06)
    first, trace = search(prior, objective, settings, 91)
    second, _ = search(prior, objective, settings, 91)
    np.testing.assert_array_equal(first, second)
    assert len(calls) > 2 * settings["population"]
    assert trace[-1]["objective"] < trace[0]["objective"]
    assert float(kl_prior(torch.tensor([1., 0.]), torch.tensor([.5, .5]))) > 0


def test_gate_is_detached_and_selective_gradient_reaches_evolution(config):
    src = torch.tensor([[1., 0.], [-1., 0.]], requires_grad=True)
    target = torch.tensor([[1., 0.], [.9, .1]], requires_grad=True)
    gate, _ = sample_gates(src, target, (.5, 1.5), 1e-8)
    assert not gate.requires_grad
    assert .5 <= gate[1] < gate[0] <= 1.5
    (ReverseGradient.apply(src) * 3).sum().backward()
    torch.testing.assert_close(src.grad, torch.full_like(src, -3))
    model = Forecaster(config["model"])
    classifier = DomainDiscriminator(6, 8)
    source = model(torch.rand(4, 12, 1))
    target = model(torch.rand(4, 12, 1))
    loss, _ = domain_loss(classifier, source.shared, target.shared, torch.ones(4))
    loss.backward()
    assert model.evolution.supplementary.weight.grad is None
    assert model.head.weight.grad is None
    assert model.evolution.score.weight.grad.abs().sum() > 0
    assert model.backbone.weight_ih_l0.grad.abs().sum() > 0
    assert classifier[0].weight.grad.abs().sum() > 0


def test_anchoring_compares_effective_residuals_and_excludes_head(config):
    model = Forecaster(config["model"], adapter_rank=2)
    for matrix in model.adapter.matrices.values():
        torch.nn.init.ones_(matrix.b)
    anchor = snapshot(model)
    mat = model.adapter.matrices["weight_ih_l0"]
    with torch.no_grad():
        mat.a.mul_(4)
        mat.b.div_(4)
        model.head.bias.add_(3)
    weights = strengths(config["algorithm"]["anchoring"], 50, 1000, 1e-8)
    assert weights["backbone"] > strengths(config["algorithm"]["anchoring"], 100, 1000, 1e-8)["backbone"]
    assert float(anchor_loss(model, anchor, weights)) == 0
    with torch.no_grad():
        mat.b.add_(1)
    expected = weights["adapter"]*(mat() - anchor["adapter"]["weight_ih_l0"]).square().sum()
    torch.testing.assert_close(anchor_loss(model, anchor, weights), expected)
