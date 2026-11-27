from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data.preprocess import clean, timestamps
from .models.forecaster import Forecaster, PlainLSTM
from .models.cyclenet import CycleNet, cycle_phase
from .serialization import load_weights


def load_predictor(checkpoint, device="cpu"):
    payload = load_weights(checkpoint, device)
    if payload.get("format") != "rmaf-sdatn-deployment-v1":
        raise ValueError("Use a merged final_model.pt deployment checkpoint.")
    kind = payload["kind"]
    if kind in {"patchtst", "timesnet", "cyclenet"}:
        from .evaluation.modern_baselines import modern_baseline
        predictor = modern_baseline(payload["architecture"], kind).to(device)
    elif kind in {"rmaf", "target_only", "multi_source", "dann", "mmd", "coral"}:
        model = Forecaster if kind == "rmaf" else PlainLSTM
        predictor = model(payload["model_config"]).to(device)
    else:
        raise ValueError(f"Unknown deployment architecture: {kind}")
    predictor.load_state_dict(payload["model"])
    return predictor.eval(), payload


def save_predictor(model, episode, cfg, kind, path):
    from .serialization import atomic_save
    settings = cfg["data"]
    meta = {key: episode.meta[key] for key in ("station", "capacity", "unit", "timezone", "capacity_source")}
    cleaning = {key: settings[key] for key in
                ("negative", "duplicate", "max_normalized_power", "date_format", "interval_minutes")}
    cleaning["negative"] = episode.meta.get("negative_policy", cleaning["negative"])
    cleaning["minimum_bin_coverage"] = episode.meta.get("minimum_bin_coverage", 1.0)
    architecture = {key: cfg[key] for key in ("model", "baselines", "algorithm")} if kind in {"patchtst", "timesnet", "cyclenet"} else {}
    atomic_save({"format": "rmaf-sdatn-deployment-v1", "kind": kind, "model": model.state_dict(),
                 "model_config": cfg["model"], "station": meta, "cleaning": cleaning, "architecture": architecture}, path)


@torch.inference_mode()
def predict_frame(checkpoint, frame, device="cpu", timezone=None, station=None, capacity=None, native_minutes=None):
    model, payload = load_predictor(checkpoint, device)
    meta, settings, cfg = payload["station"], payload["cleaning"], payload["model_config"]
    if station is not None and station != meta["station"]:
        raise ValueError("Input station differs from the checkpoint target.")
    if "station" in frame and set(frame["station"].astype(str)) != {meta["station"]}:
        raise ValueError("Input CSV contains station identifiers inconsistent with the checkpoint.")
    if capacity is not None and not np.isclose(capacity, meta["capacity"]):
        raise ValueError("Input capacity differs from the checkpoint capacity.")
    zone = timezone or meta["timezone"]
    if not {"timestamp", "power"} <= set(frame.columns):
        raise ValueError("Prediction CSV requires timestamp and power columns in the checkpoint power unit.")
    prepared, _ = clean(frame, meta["capacity"], settings, zone, native_minutes=native_minutes,
                        minimum_bin_coverage=settings.get("minimum_bin_coverage", 1.0))
    interval = pd.Timedelta(minutes=settings["interval_minutes"])
    recent = timestamps(frame.timestamp, zone, settings["date_format"]).max()
    if native_minutes is not None and native_minutes != settings["interval_minutes"]:
        recent = recent.floor(f"{settings['interval_minutes']}min")
        prepared = prepared[prepared.index <= recent]
    history = prepared.tail(cfg["history"])
    if (len(history) < cfg["history"] or history.index[-1] != recent
            or not (history.index[1:] - history.index[:-1] == interval).all()):
        raise ValueError("Latest history must end at the latest input timestamp with consecutive valid observations.")
    x = torch.as_tensor(history.normalized_power.to_numpy(np.float32), device=device)[None, :, None]
    context = []
    if isinstance(model, CycleNet):
        context.append(cycle_phase(history.index[:1], model.period, model.interval).to(device))
    result = model(x, *context).power[0].cpu().numpy()
    dates = pd.date_range(recent + interval, periods=cfg["horizon"], freq=interval).tz_convert(zone)
    return pd.DataFrame({"timestamp": dates, "step": np.arange(1, len(dates)+1), "station": meta["station"],
                         "prediction_normalized": result, "prediction_power": result * meta["capacity"], "unit": meta["unit"]})


def predict_csv(checkpoint, input_path, output_path=None, **options):
    result = predict_frame(checkpoint, pd.read_csv(input_path, dtype={"station": str}), **options)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
    return result
