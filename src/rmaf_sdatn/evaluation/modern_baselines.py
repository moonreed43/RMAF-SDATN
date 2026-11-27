"""Power-only adaptations of PatchTST and TimesNet forecasting blocks."""
import math

import torch
from torch import nn

from ..models.forecaster import Forecast


def reversible_normalize(x, epsilon):
    mean = x.mean(dim=1, keepdim=True).detach()
    scale = (x.var(dim=1, keepdim=True, unbiased=False) + epsilon).sqrt().detach()
    return (x-mean)/scale, mean.squeeze(-1), scale.squeeze(-1)


class PatchTST(nn.Module):
    """Univariate patch tokens, learnable position, Transformer and flatten head."""
    def __init__(self, model, cfg, epsilon):
        super().__init__()
        self.patch = cfg["patch_length"]
        self.stride = cfg["patch_stride"]
        self.epsilon = epsilon
        tokens = (model["history"] - self.patch) // self.stride + 2
        width = cfg["width"]
        self.embed = nn.Linear(self.patch, width)
        self.position = nn.Parameter(torch.zeros(1, tokens, width))
        nn.init.uniform_(self.position, -0.02, 0.02)
        block = nn.TransformerEncoderLayer(width, cfg["heads"], dim_feedforward=cfg["feedforward"],
                                           dropout=model["dropout"], activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(block, cfg["layers"])
        self.head = nn.Linear(width*tokens, model["horizon"])

    def forward(self, x):
        x, mean, scale = reversible_normalize(x, self.epsilon)
        x = torch.cat([x, x[:, -1:].expand(-1, self.stride, -1)], dim=1)
        tokens = x.squeeze(-1).unfold(1, self.patch, self.stride)
        features = self.encoder(self.embed(tokens) + self.position).flatten(1)
        power = (self.head(features)*scale + mean).clamp_min(0)
        return Forecast(power, features, None, None, None)


class Inception2D(nn.Module):
    def __init__(self, inputs, outputs, kernels):
        super().__init__()
        self.kernels = nn.ModuleList([nn.Conv2d(inputs, outputs, 2*i+1, padding=i) for i in range(kernels)])

    def forward(self, x):
        return torch.stack([kernel(x) for kernel in self.kernels], dim=0).mean(0)


class TimesBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.top_k = cfg["periods"]
        self.conv = nn.Sequential(Inception2D(cfg["width"], cfg["feedforward"], cfg["kernels"]), nn.GELU(),
                                  Inception2D(cfg["feedforward"], cfg["width"], cfg["kernels"]))

    def forward(self, x):
        batch, length, width = x.shape
        amplitude = torch.fft.rfft(x, dim=1).abs()
        freq = amplitude.mean(dim=(0, 2))[1:].topk(self.top_k).indices + 1
        outputs = []
        for index in freq.tolist():
            period = length // index
            padded = math.ceil(length/period)*period
            arr = nn.functional.pad(x, (0, 0, 0, padded-length))
            arr = arr.reshape(batch, padded//period, period, width).permute(0, 3, 1, 2)
            outputs.append(self.conv(arr).permute(0, 2, 3, 1).reshape(batch, padded, width)[:, :length])
        weights = amplitude.mean(dim=-1)[:, freq].softmax(dim=1)
        return x + (torch.stack(outputs, dim=-1) * weights[:, None, None]).sum(dim=-1)


class TimesNet(nn.Module):
    """Dominant FFT periods, intraperiod/interperiod grids and 2D residual blocks."""
    def __init__(self, model, cfg, epsilon):
        super().__init__()
        history, self.horizon = model["history"], model["horizon"]
        self.epsilon = epsilon
        width = cfg["width"]
        self.embed = nn.Conv1d(1, width, 3, padding=1, padding_mode="circular", bias=False)
        positions = torch.arange(history)[:, None]
        angles = positions * torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0)/width))
        positional = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)[:, :width]
        self.register_buffer("position", positional)
        self.dropout = nn.Dropout(model["dropout"])
        self.extend = nn.Linear(history, history+self.horizon)
        self.blocks = nn.ModuleList([TimesBlock(cfg) for _ in range(cfg["layers"])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        x, mean, scale = reversible_normalize(x, self.epsilon)
        feature = self.dropout(self.embed(x.transpose(1, 2)).transpose(1, 2) + self.position)
        feature = self.extend(feature.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            feature = self.norm(block(feature))
        power = self.head(feature[:, -self.horizon:]).squeeze(-1)
        return Forecast((power*scale+mean).clamp_min(0), feature[:, -1], None, None, None)


def modern_baseline(config, method):
    from ..models.cyclenet import CycleNet
    models = {"patchtst": PatchTST, "timesnet": TimesNet, "cyclenet": CycleNet}
    if method not in models:
        raise ValueError(f"Unsupported power-only architecture: {method}")
    return models[method](config["model"], config["baselines"][method], config["algorithm"]["epsilon"])
