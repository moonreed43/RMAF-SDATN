from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from rmaf_sdatn.inference import predict_csv, save_predictor
from rmaf_sdatn.models.forecaster import Forecaster


def test_deployment_csv_roundtrip_has_no_adapter_dependency(tmp_path, config):
    model = Forecaster(config["model"], adapter_rank=2).eval()
    station = SimpleNamespace(meta={"station": "X", "capacity": 20.0, "unit": "MW", "timezone": "UTC",
                                    "capacity_source": "test metadata"})
    checkpoint = tmp_path/"model.pt"
    save_predictor(model.merged(), station, config, "rmaf", checkpoint)
    history = pd.DataFrame({"timestamp": pd.date_range("2020-05-01", periods=12, freq="15min", tz="UTC"),
                            "power": np.linspace(0, 10, 12), "station": "X"})
    source = tmp_path/"history.csv"
    history.to_csv(source, index=False)
    result = predict_csv(checkpoint, source, tmp_path/"output.csv", capacity=20)
    assert result.timestamp.iloc[0] == history.timestamp.iloc[-1] + pd.Timedelta(minutes=15)
    x = torch.from_numpy(history.power.to_numpy(np.float32)/20).view(1, 12, 1)
    np.testing.assert_allclose(result.prediction_power, model(x).power.detach().numpy().ravel()*20, rtol=1e-5)
    history.drop(index=10).to_csv(source, index=False)
    with pytest.raises(ValueError, match="consecutive"):
        predict_csv(checkpoint, source)
