from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset, TensorDataset


@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray
    first: pd.DatetimeIndex
    last: pd.DatetimeIndex
    history_start: pd.DatetimeIndex

    def __len__(self):
        return len(self.x)

    def between(self, start, end, allow_history=True):
        mask = (self.first >= start) & (self.last < end)
        if not allow_history:
            mask &= self.history_start >= start
        return Windows(self.x[mask], self.y[mask], self.first[mask], self.last[mask], self.history_start[mask])

    def labeled(self):
        return TensorDataset(torch.from_numpy(self.x), torch.from_numpy(self.y))

    def unlabeled(self):
        return InputsOnly(self.x)


class InputsOnly(Dataset):
    def __init__(self, x):
        self.x = torch.from_numpy(x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


def window_power(power, history, horizon, interval):
    length = history + horizon
    index = pd.DatetimeIndex(power.index)
    values = power.to_numpy(dtype=np.float32)
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Window construction requires sorted, unique timestamps.")
    if len(index) < length:
        empty = index[:0]
        return Windows(np.empty((0, history, 1), np.float32), np.empty((0, horizon), np.float32), empty, empty, empty)
    times = index.to_numpy(dtype="datetime64[ns]")
    contiguous = times[length-1:] - times[:1-length] == np.timedelta64((length-1)*interval, "m")
    contiguous &= sliding_window_view(np.diff(times) == np.timedelta64(interval, "m"), length-1).all(axis=1)
    view = sliding_window_view(values, length)
    valid = contiguous & np.isfinite(view).all(axis=1)
    rows = view[valid]
    return Windows(rows[:, :history, None].copy(), rows[:, history:].copy(), index[history:len(index)-horizon+1][valid],
                   index[length-1:][valid], index[:len(valid)][valid])
