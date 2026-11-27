import numpy as np
import pandas as pd
import torch

from ..models.cyclenet import prediction_inputs


def regression(y, prediction):
    y, prediction = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != prediction.shape:
        raise ValueError(f"Prediction and label shapes differ: {prediction.shape} versus {y.shape}.")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Forecast metrics require finite observations and predictions.")
    n = y.size
    if not n:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    residual = prediction - y
    variance = np.square(y - y.mean()).sum()
    return {"n": n, "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "mae": float(np.abs(residual).mean()), "r2": float(1 - np.square(residual).sum() / variance) if variance else np.nan}


@torch.inference_mode()
def predict_batches(model, tr, windows):
    if not len(windows):
        raise ValueError("No valid windows are available for evaluation.")
    if model is None:
        return np.repeat(windows.x[:, -1, :], windows.y.shape[-1], axis=1)
    model.eval()
    return torch.cat([model(*(value.to(tr.device) for value in batch)).power.cpu()
                      for batch in tr.loader(prediction_inputs(windows, model))]).numpy()


def score_predictions(windows, pred, episode, cfg, directory):
    pred = np.asarray(pred)
    if pred.shape != windows.y.shape or not np.isfinite(pred).all():
        raise ValueError("Predictions must be finite and match the target window/output shape.")
    horizon = windows.y.shape[1]
    interval = pd.Timedelta(minutes=cfg["data"]["interval_minutes"])
    capacity = episode.meta["capacity"]
    results = []
    threshold = np.quantile(episode.source_ramps, cfg["evaluation"]["ramp_quantile"]) if len(episode.source_ramps) else np.nan
    ramp = (np.abs(windows.y[:, 0]-windows.x[:, -1, 0]) >= threshold) if horizon == 1 else None
    tables = []
    for step in range(1, horizon+1):
        current = windows.y[:, step-1]
        estimate = pred[:, step-1]
        ts = windows.first + (step-1)*interval
        table = pd.DataFrame({"timestamp": ts, "origin": windows.first-interval, "step": step, "actual": current,
                              "prediction": estimate, "actual_power": current*capacity, "prediction_power": estimate*capacity,
                              "unit": episode.meta["unit"]})
        if ramp is not None and np.isfinite(threshold):
            table["strong_ramp"] = ramp
        tables.append(table)
        if step not in cfg["evaluation"].get("steps", list(range(1, horizon+1))):
            continue
        select = {"all": np.ones(len(current), dtype=bool),
                  "active": current > cfg["evaluation"]["active_threshold"]}
        if ramp is not None and np.isfinite(threshold):
            select.update(ramp=ramp, non_ramp=~ramp)
        for subset, mask in select.items():
            results.append({"step": step, "subset": subset, **regression(current[mask], estimate[mask])})
    pd.concat(tables, ignore_index=True).to_csv(directory/"predictions.csv", index=False)
    return results, float(threshold)
