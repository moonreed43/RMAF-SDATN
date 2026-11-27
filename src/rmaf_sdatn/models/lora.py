import math

import torch
from torch import nn


class LowRankMatrix(nn.Module):
    def __init__(self, rows, cols, rank, scale):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, cols))
        self.b = nn.Parameter(torch.zeros(rows, rank))
        self.scale = scale
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self):
        return self.scale * (self.b @ self.a)


class LoRA(nn.Module):
    def __init__(self, backbone, rank, alpha):
        super().__init__()
        self.rank = rank
        self.matrices = nn.ModuleDict({
            name: LowRankMatrix(*weight.shape, rank, alpha / rank)
            for name, weight in backbone.named_parameters() if name.startswith(("weight_ih_", "weight_hh_"))
        })

    def residuals(self):
        return {name: matrix() for name, matrix in self.matrices.items()}

    @classmethod
    @torch.no_grad()
    def fuse(cls, backbone, adapters, weights):
        rank = sum(a.rank for a in adapters)
        fused = cls(backbone, rank, rank).to(next(backbone.parameters()))
        for name, matrix in fused.matrices.items():
            source = [a.matrices[name] for a in adapters]
            matrix.a.copy_(torch.cat([m.a for m in source], dim=0))
            matrix.b.copy_(torch.cat([m.b * (float(w) * m.scale) for w, m in zip(weights, source)], dim=1))
        return fused


def weighted_residuals(stacked, weights):
    return {name: torch.einsum("k,kij->ij", weights, matrices) for name, matrices in stacked.items()}
