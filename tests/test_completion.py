import json
import random
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rmaf_sdatn.algorithms.fusion import search
from rmaf_sdatn.algorithms.reliability import FeatureMoments, covariance
from rmaf_sdatn.data.transfers import ranged_download
from rmaf_sdatn.data.preprocess import sha256, timestamps
from rmaf_sdatn.data.splits import StationData
from rmaf_sdatn.config import load_config
from rmaf_sdatn.experiments import recover_results
from rmaf_sdatn.inference import predict_frame, save_predictor
from rmaf_sdatn.models.cyclenet import CycleNet, cycle_phase
from rmaf_sdatn.serialization import write_json
from rmaf_sdatn.validation import validate_config
from rmaf_sdatn.locking import file_lock
from rmaf_sdatn.evaluation.metrics import regression


def test_search_resume_matches_uninterrupted_population(config, tmp_path):
    settings = dict(config["algorithm"]["fusion"], population=8, generations=5)
    prior = np.array([.2, .3, .5])
    target = np.array([.4, .2, .4])
    expected, trace = search(prior, lambda r: np.square(r-target).sum(), settings, 11)
    counter = 0

    def interrupted(vector):
        nonlocal counter
        counter += 1
        if counter == 11:
            raise InterruptedError
        return np.square(vector-target).sum()

    checkpoint = tmp_path/"search.pt"
    with pytest.raises(InterruptedError):
        search(prior, interrupted, settings, 11, checkpoint)
    actual, resumed_trace = search(prior, lambda r: np.square(r-target).sum(), settings, 11, checkpoint)
    np.testing.assert_array_equal(expected, actual)
    assert trace == resumed_trace


def test_streaming_covariance_preserves_station_statistics():
    features = torch.randn(37, 9, dtype=torch.float64) + 5000
    moments = FeatureMoments()
    for batch in features.split(7):
        moments.update(batch)
    mean, cov = moments.statistics()
    torch.testing.assert_close(mean, features.mean(0), atol=1e-10, rtol=0)
    torch.testing.assert_close(cov, covariance(features), atol=1e-10, rtol=0)


def test_cycle_phase_and_deployment_roundtrip(config, tmp_path):
    from types import SimpleNamespace
    config["model"]["horizon"] = 24
    model = CycleNet(config["model"], config["baselines"]["cyclenet"], 1e-8).eval()
    with torch.no_grad():
        model.cycle.copy_(torch.linspace(0, .1, model.period))
    dates = pd.date_range("2021-04-02", periods=12, freq="15min", tz="Asia/Shanghai")
    phase = cycle_phase(dates[:1], model.period, 15)
    assert phase.item() == 64
    frame = pd.DataFrame({"timestamp": dates, "power": np.linspace(.5, 1.5, 12)})
    episode = SimpleNamespace(meta={"station": "X", "capacity": 2., "unit": "MW",
                                    "timezone": "Asia/Shanghai", "capacity_source": "test"})
    checkpoint = tmp_path/"cycle.pt"
    save_predictor(model, episode, config, "cyclenet", checkpoint)
    x = torch.from_numpy(frame.power.to_numpy(np.float32)/2)[None, :, None]
    expected = model(x, phase).power
    expected.mean().backward()
    assert model.cycle.grad.abs().sum() > 0
    result = predict_frame(checkpoint, frame)
    assert len(result) == 24
    np.testing.assert_allclose(result.prediction_normalized, expected.detach().numpy()[0], rtol=1e-5)


@pytest.mark.parametrize("field,value", [("discriminator", "typo"), ("gate", "bad"), ("branch", "mixed")])
def test_configuration_does_not_silently_accept_unknown_modes(config, field, value):
    config["algorithm"]["alignment"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(config)


def test_recovery_of_completed_trials_is_idempotent(tmp_path):
    record = {"target": "H7", "window": "final", "transfer": "mixed", "budget": 30, "seed": 42,
              "method": "rmaf", "variant": "full", "source_case": "all", "path": str(tmp_path/"trial"),
              "incompatible_count": None, "training_seconds": 1.0, "single_sample_latency_ms": 1.0}
    metric = {"step": 1, "subset": "all", "rmse": .1, "mae": .05, "r2": .9}
    write_json({"record": record, "metrics": [metric]}, tmp_path/"trial/experiment_record.json")
    recovered = recover_results(tmp_path)
    assert len(recovered) == 1 and recovered.rmse.iloc[0] == .1
    assert len(recover_results(tmp_path)) == 1
    assert (tmp_path/"mean_sd.csv").exists()


def test_partial_range_transfer_recovers_verified_parts(tmp_path):
    payload = bytes(range(256))*200
    fail = {"enabled": True}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("ETag", '"fixed-test-object"')
            self.end_headers()

        def do_GET(self):
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            if start == 0 and fail["enabled"]:
                self.send_error(503)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("ETag", '"fixed-test-object"')
            self.end_headers()
            self.wfile.write(payload[start:end+1])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/object"
        file = tmp_path/"download.bin"
        with pytest.raises(RuntimeError):
            ranged_download(url, file, len(payload), workers=2, chunk_bytes=8192)
        assert not file.exists()
        assert list(tmp_path.glob("*.parts/*.bin"))
        partial = file.with_suffix(".bin.parts")/"000000000000.tmp"
        partial.write_bytes(payload[:4096])
        fail["enabled"] = False
        ranged_download(url, file, len(payload), workers=2, chunk_bytes=8192)
        assert file.read_bytes() == payload
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_cli_import_uses_this_checkout():
    result = subprocess.run([sys.executable, "scripts/run_experiments.py", "--help"],
                            capture_output=True, text=True, check=True)
    assert "--budgets" in result.stdout


def test_config_include_cycle_is_reported(tmp_path):
    first, second = tmp_path/"first.yaml", tmp_path/"second.yaml"
    first.write_text("include: [second.yaml]\n", encoding="utf-8")
    second.write_text("include: [first.yaml]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Circular YAML include"):
        load_config(first)


def test_processed_file_integrity_and_capacity_are_verified(config, tmp_path):
    config["data"]["root"] = str(tmp_path)
    folder = tmp_path/"processed"
    folder.mkdir()
    path = folder/"H7.csv"
    pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=32, freq="15min", tz="UTC"),
                  "power": np.ones(32), "normalized_power": np.ones(32)/2}).to_csv(path, index=False)
    metadata = {"interval_minutes": 15, "capacity": 2.0, "preprocessing_version": 2,
                "processed_sha256": sha256(path)}
    write_json(metadata, path.with_suffix(".json"))
    assert len(StationData(config).get("H7")[1]) == 20
    write_json({**metadata, "capacity": 3.}, path.with_suffix(".json"))
    with pytest.raises(ValueError, match="power/capacity"):
        StationData(config).get("H7")
    write_json(metadata, path.with_suffix(".json"))
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="differs from its quality record"):
        StationData(config).get("H7")


def test_mixed_naive_and_offset_timestamps_do_not_change_timezone_silently():
    with pytest.raises(ValueError, match="mixes timezone-aware and naive"):
        timestamps(pd.Series(["2021-01-01 00:00", "2021-01-01T00:15:00Z"]), "Asia/Shanghai")


def test_writer_lock_is_exclusive_and_released_after_error(tmp_path):
    path = tmp_path/"run.lock"
    with pytest.raises(InterruptedError):
        with file_lock(path):
            with pytest.raises(RuntimeError, match="another process"):
                with file_lock(path):
                    pytest.fail("Concurrent writer acquired the lock.")
            raise InterruptedError
    with file_lock(path):
        assert path.exists()


def test_metrics_reject_broadcasting_and_nonfinite_predictions():
    with pytest.raises(ValueError, match="shapes differ"):
        regression(np.ones((3, 1)), np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        regression(np.ones(3), np.array([1., 2., np.nan]))


def test_fine_cadence_inference_uses_only_completed_intervals(config, tmp_path):
    from types import SimpleNamespace
    from rmaf_sdatn.models.forecaster import Forecaster
    model = Forecaster(config["model"]).eval()
    episode = SimpleNamespace(meta={"station": "X", "capacity": 2., "unit": "MW",
                                    "timezone": "UTC", "capacity_source": "test"})
    path = tmp_path/"model.pt"
    save_predictor(model, episode, config, "rmaf", path)
    frame = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=39, freq="5min", tz="UTC"),
                          "power": np.arange(39, dtype=float)*2/39})
    prediction = predict_frame(path, frame, native_minutes=5)
    assert prediction.timestamp.iloc[0] == pd.Timestamp("2021-01-01 03:15Z")
    x = torch.tensor(np.arange(2, 36, 3)/39, dtype=torch.float32)[None, :, None]
    expected = model(x).power.detach().numpy()[0]
    np.testing.assert_allclose(prediction.prediction_normalized, expected, rtol=1e-5)


def test_batch_data_failures_can_be_recorded_and_resolved(config, tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("rmaf_sdatn.experiments")
    config["experiment"].update(targets=["H7", "H8"], output=str(tmp_path), name="batch", keep_going=True)
    visits = []

    def unavailable(store, station, failures=None):
        visits.append(station)
        if station == "H7":
            raise ValueError("Insufficient observed power coverage.")
        return {}

    monkeypatch.setattr(module, "test_windows", unavailable)
    root = module.experiments(config)
    assert visits == ["H7", "H8"]
    failures = json.loads((root/"failures.json").read_text(encoding="utf-8"))
    assert len(failures) == 1 and failures[0]["context"] == {"target": "H7"}
    monkeypatch.setattr(module, "test_windows", lambda store, station, failures=None: {})
    module.experiments(config)
    assert json.loads((root/"failures.json").read_text(encoding="utf-8")) == []


def test_sparse_external_season_does_not_hide_clean_seasons(config):
    from rmaf_sdatn.data.splits import test_windows as seasonal_windows
    from rmaf_sdatn.data.windows import window_power
    config["data"]["pvdaq"]["years"] = [2023]
    dates = pd.date_range("2023-05-01", "2023-11-01", freq="15min", tz="UTC", inclusive="left")
    power = pd.Series(.2, index=dates)
    windows = window_power(power, 12, 1, 15)

    class Store:
        cfg = config

        def get(self, station):
            return power, windows, {"dataset": "pvdaq", "timezone": "UTC"}

    with pytest.raises(ValueError, match="2023-03-01"):
        seasonal_windows(Store(), "X")
    failures = {}
    seasons = seasonal_windows(Store(), "X", failures=failures)
    assert list(failures) == ["2023-03-01"]
    assert len(seasons) == 3
    dec_start, dec_end, dev_end = seasons["2023-12-01"]
    assert dec_end <= pd.Timestamp("2023-11-01", tz="UTC")
    assert dec_start >= seasons["2023-09-01"][1]
    assert dev_end <= seasons["2023-09-01"][0]


def test_failed_reprocessing_does_not_silently_reuse_an_older_dataset(config, tmp_path):
    from rmaf_sdatn.data.preprocess import preprocess
    data = config["data"]
    data["root"] = str(tmp_path/"data")
    data["capacity_overrides"] = str(tmp_path/"no_overrides.yaml")
    raw = tmp_path/"observations.csv"
    data["stations"]["H7"].update(files=[str(raw)], timestamp_column="timestamp", power_column="power",
                                 input_timezone="UTC", capacity={"value": 2., "unit": "MW", "source": "test"})
    frame = pd.DataFrame({"timestamp": pd.date_range("2018-08-01", periods=32, freq="15min", tz="UTC"),
                          "power": np.ones(32)})
    frame.to_csv(raw,index=False)
    preprocess("H7",data,config["model"])
    assert len(StationData(config).get("H7")[1]) == 20
    frame.assign(power=-1).to_csv(raw,index=False)
    with pytest.raises(ValueError, match="no clean observations"):
        preprocess("H7",data,config["model"])
    with pytest.raises(ValueError, match="latest preprocessing attempt failed"):
        StationData(config).get("H7")
    frame.to_csv(raw,index=False)
    preprocess("H7",data,config["model"])
    assert len(StationData(config).get("H7")[1]) == 20
