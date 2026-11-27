import torch


def covariance(features):
    centered = features - features.mean(dim=0)
    return centered.T @ centered / max(len(features) - 1, 1)


class FeatureMoments:
    """Merge minibatch moments without retaining the station's feature matrix."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.centered_sum = None

    def update(self, features):
        values = features.detach().double()
        if len(values) == 0:
            return
        mean = values.mean(0)
        centered = values - mean
        scatter = centered.T @ centered
        if self.count == 0:
            self.count, self.mean, self.centered_sum = len(values), mean, scatter
            return
        total = self.count + len(values)
        difference = mean - self.mean
        self.centered_sum = self.centered_sum + scatter + torch.outer(difference, difference) * self.count * len(values) / total
        self.mean = self.mean + difference * (len(values)/total)
        self.count = total

    def statistics(self):
        if self.count == 0:
            raise ValueError("Cannot estimate moments from an empty station.")
        return self.mean, self.centered_sum/max(self.count-1, 1)


def discrepancy_statistics(source, target, beta):
    source_mean, source_covariance = source
    target_mean, target_covariance = target
    return ((source_mean-target_mean).square().sum() + beta*(source_covariance-target_covariance).square().sum())


def discrepancy(source, target, beta):
    source, target = source.double(), target.double()
    return ((source.mean(0) - target.mean(0)).square().sum()
            + beta * (covariance(source) - covariance(target)).square().sum())


def minmax(x, epsilon):
    return (x - x.min()) / (x.max() - x.min()).clamp_min(epsilon)


def reliability(delta, error, eta, temperature, epsilon):
    if not 0 <= eta <= 1 or temperature <= 0:
        raise ValueError("Reliability requires eta in [0,1] and positive temperature.")
    norm_delta = minmax(delta.double(), epsilon)
    norm_error = minmax(error.double(), epsilon)
    scores = -(eta * norm_error + (1-eta) * norm_delta) / temperature
    positive = scores.softmax(dim=0).clamp_min(torch.finfo(scores.dtype).tiny)
    if not torch.isfinite(positive).all():
        raise ValueError("Nonfinite reliability scores.")
    return positive / positive.sum(), norm_delta, norm_error
