from copy import deepcopy
from typing import NamedTuple

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F

from .lora import LoRA


class Forecast(NamedTuple):
    power: torch.Tensor
    shared: torch.Tensor
    supplementary: torch.Tensor
    attention: torch.Tensor
    magnitudes: torch.Tensor


class Evolution(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg["hidden"]
        self.scales = tuple(cfg["scales"])
        if not self.scales or any(s <= 0 or s >= cfg["history"] for s in self.scales):
            raise ValueError("Each evolution lag must be positive and shorter than the input.")
        self.embedding = nn.Linear(h * (1 + len(self.scales)) + len(self.scales), cfg["embedding"])
        self.score = nn.Linear(cfg["embedding"], 1)
        self.shared = nn.Linear(h, cfg["shared_dim"], bias=False)
        self.supplementary = nn.Linear(h, cfg["supplementary_dim"], bias=False)

    def forward(self, h):
        changes = [torch.cat((torch.zeros_like(h[:, :s]), h[:, s:] - h[:, :-s]), dim=1) for s in self.scales]
        magnitudes = torch.stack([torch.linalg.vector_norm(delta, dim=-1) for delta in changes], dim=-1)
        descriptor = torch.cat([h, *changes, magnitudes], dim=-1)
        attention = self.score(torch.tanh(self.embedding(descriptor))).squeeze(-1).softmax(dim=1)
        shared = (attention.unsqueeze(-1) * self.shared(h)).sum(dim=1)
        return shared, self.supplementary(h[:, -1]), attention, magnitudes


class Forecaster(nn.Module):
    def __init__(self, cfg, adapter_rank=0):
        super().__init__()
        self.cfg = dict(cfg)
        self.backbone = nn.LSTM(1, cfg["hidden"], cfg["layers"], batch_first=True,
                                dropout=cfg["dropout"] if cfg["layers"] > 1 else 0)
        self.evolution = Evolution(cfg)
        self.head = nn.Linear(cfg["shared_dim"] + cfg["supplementary_dim"], cfg["horizon"])
        self.adapter = LoRA(self.backbone, adapter_rank, adapter_rank) if adapter_rank else None

    def hidden(self, x, residuals=None):
        if residuals is None and self.adapter is not None:
            residuals = self.adapter.residuals()
        if residuals is None:
            return self.backbone(x)[0]
        weights = dict(self.backbone.named_parameters())
        effective = {key: weights[key] + delta for key, delta in residuals.items()}
        return functional_call(self.backbone, effective, (x,))[0]

    def forward(self, x, residuals=None):
        shared, supplementary, attention, magnitudes = self.evolution(self.hidden(x, residuals))
        power = F.softplus(self.head(torch.cat((shared, supplementary), dim=-1)))
        return Forecast(power, shared, supplementary, attention, magnitudes)

    @torch.no_grad()
    def merged(self):
        device_param = next(self.parameters())
        # Build a fresh LSTM. Functional calls may leave transient flat-weight references in cuDNN.
        deployed = Forecaster(self.cfg).to(device_param)
        deployed.load_state_dict({k: v for k, v in self.state_dict().items() if not k.startswith("adapter.")})
        if self.adapter is not None:
            for name, residual in self.adapter.residuals().items():
                getattr(deployed.backbone, name).add_(residual)
        return deployed.eval()


class PlainLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.backbone = nn.LSTM(1, cfg["hidden"], cfg["layers"], batch_first=True,
                                dropout=cfg["dropout"] if cfg["layers"] > 1 else 0)
        self.head = nn.Linear(cfg["hidden"], cfg["horizon"])

    def forward(self, x):
        features = self.backbone(x)[0][:, -1]
        return Forecast(F.softplus(self.head(features)), features, None, None, None)


class DomainDiscriminator(nn.Sequential):
    def __init__(self, dim, hidden):
        super().__init__(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
