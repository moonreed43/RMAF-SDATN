import math

import torch
from torch.nn import functional as F

from .reliability import covariance


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


def adversarial_strength(progress, maximum, gamma):
    return maximum * math.tanh(gamma * progress / 2)


@torch.no_grad()
def sample_gates(source, target, bounds, epsilon):
    source = source / (source.norm(dim=-1, keepdim=True) + epsilon)
    target = target / (target.norm(dim=-1, keepdim=True) + epsilon)
    distances = torch.linalg.vector_norm(source[:, None] - target[None], dim=-1)
    sigma = distances.flatten().quantile(0.5).clamp_min(epsilon)
    similarity = torch.exp(-0.5 * (distances / sigma).square()).mean(dim=1)
    gates = (similarity / (similarity.mean() + epsilon)).clamp(*bounds)
    return gates, similarity


def alignment_features(result, branch):
    if branch == "both":
        return torch.cat((result.shared, result.supplementary), dim=-1)
    return result.shared


def domain_loss(discriminator, source, target, gates, mode="online", fraction=0.5):
    source_logit = discriminator(ReverseGradient.apply(source)).flatten()
    target_logit = discriminator(ReverseGradient.apply(target)).flatten()
    source_bce = F.softplus(-source_logit)
    if mode in {"high", "low"}:
        k = max(1, round(len(gates) * fraction))
        selected = gates.argsort(descending=(mode == "high"), stable=True)[:k]
        source_bce = source_bce[selected].mean()
    else:
        source_bce = (source_bce * gates).mean()
    target_bce = F.softplus(target_logit).mean()
    with torch.no_grad():
        balanced_accuracy = ((source_logit >= 0).float().mean() + (target_logit < 0).float().mean()) / 2
    return source_bce + target_bce, balanced_accuracy


def mmd(source, target, epsilon):
    distances = torch.cdist(torch.cat([source, target]), torch.cat([source, target])).square()
    with torch.no_grad():
        bandwidth = torch.cdist(source, target).flatten().quantile(0.5).square().clamp_min(epsilon)
    kernel = torch.exp(-distances / (2 * bandwidth))
    n = len(source)
    return kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean()


def coral(source, target):
    return (covariance(source) - covariance(target)).square().sum() / (4 * source.shape[-1] ** 2)
