"""CycleNet residual cycle forecasting for capacity-normalized PV power.

Method reference: https://github.com/ACAT-SCUT/CycleNet
Phase specifies the first historical observation, counted on the UTC sampling grid.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset

from .forecaster import Forecast


def cycle_phase(index, period, interval_minutes):
    timestamps = pd.DatetimeIndex(index)
    if timestamps.tz is None:
        raise ValueError("Cycle phase requires timezone-aware timestamps.")
    ticks = timestamps.tz_convert("UTC").as_unit("ns").asi8 // (interval_minutes * 60 * 10**9)
    return torch.from_numpy(np.asarray(ticks % period, dtype=np.int64).copy())


class CycleNet(nn.Module):
    def __init__(self, model, settings, epsilon):
        super().__init__()
        self.history = model["history"]
        self.horizon = model["horizon"]
        self.period = settings["period"]
        self.interval = settings["interval_minutes"]
        self.revin = settings["revin"]
        self.epsilon = epsilon
        self.cycle = nn.Parameter(torch.zeros(self.period))
        if settings["head"] == "linear":
            self.residual = nn.Linear(self.history, self.horizon)
        else:
            self.residual = nn.Sequential(nn.Linear(self.history, settings["width"]), nn.ReLU(),
                                          nn.Linear(settings["width"], self.horizon))

    def forward(self, x, phase):
        values = x.squeeze(-1)
        mean = values.mean(1, keepdim=True) if self.revin else values.new_zeros(len(x), 1)
        scale = (values.var(1, keepdim=True, unbiased=True) + self.epsilon).sqrt() if self.revin else torch.ones_like(mean)
        offsets = phase[:, None] + torch.arange(self.history + self.horizon, device=x.device)
        cycle = self.cycle[offsets.remainder(self.period)]
        features = (values - mean) / scale - cycle[:, :self.history]
        prediction = self.residual(features) + cycle[:, self.history:]
        return Forecast((prediction * scale + mean).clamp_min(0), features, None, None, None)


def supervised_dataset(windows, model):
    tensors = [torch.from_numpy(windows.x), torch.from_numpy(windows.y)]
    if isinstance(model, CycleNet):
        tensors.append(cycle_phase(windows.history_start, model.period, model.interval))
    return TensorDataset(*tensors)


def prediction_inputs(windows, model):
    tensors = [torch.from_numpy(windows.x)]
    if isinstance(model, CycleNet):
        tensors.append(cycle_phase(windows.history_start, model.period, model.interval))
    return TensorDataset(*tensors)
