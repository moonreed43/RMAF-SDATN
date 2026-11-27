import numpy as np
import pandas as pd

from rmaf_sdatn.data.preprocess import clean
from rmaf_sdatn.data.splits import budget_slots, build_episode, source_ids
from rmaf_sdatn.data.windows import window_power


def test_clean_preserves_zero_and_ignores_duplicate_and_off_grid(config):
    frame = pd.DataFrame({"timestamp": ["2019-07-01 00:15", "2019-07-01 00:00", "2019-07-01 00:15",
                                        "2019-07-01 00:30", "bad", "2019-07-01 00:45", "2019-07-01 00:51"],
                          "power": [20, 0, 40, -1, 6, np.nan, 50]})
    result, stats = clean(frame, 100, config["data"], "Asia/Shanghai")
    np.testing.assert_allclose(result.normalized_power, [0, .3])
    assert result.index[0] == pd.Timestamp("2019-06-30 16:00Z")
    assert stats["excluded_rows"] == 4
    assert stats["duplicate_rows_combined"] == 1


def test_labels_do_not_straddle_boundaries_and_history_is_causal():
    idx = pd.date_range("2020-06-01", periods=96, freq="15min", tz="UTC").delete(31)
    win = window_power(pd.Series(np.arange(len(idx)), index=idx), 12, 4, 15)
    train = win.between(idx[0], idx[48])
    test = win.between(idx[48], idx[-1] + pd.Timedelta(minutes=15))
    assert train.x.shape[1:] == (12, 1)
    assert test.y.shape[1] == 4
    assert train.last.max() < test.first.min()
    assert (win.first - win.history_start == pd.Timedelta(hours=3)).all()
    assert all(not (a < idx[31] <= b) for a, b in zip(win.history_start, win.last))
    assert test.history_start.min() < idx[48]
    assert test.unlabeled()[0].shape == (12, 1)


def test_budgets_and_regional_folds(config):
    data = config["data"]
    assert list(budget_slots(30, data["ratios"], 15, 2)) == [384, 384, 1536, 576]
    for budget in [1, 3, 7, 14, 30]:
        slots = budget_slots(budget, data["ratios"], 15, 2)
        assert slots.sum() == budget*96 and slots.min() >= 2
    for target in config["experiment"]["targets"]:
        mixed = set(source_ids(config, target, "mixed"))
        intra = set(source_ids(config, target, "intra"))
        inter = set(source_ids(config, target, "inter"))
        assert target not in mixed and len(mixed) == 7
        assert mixed == intra | inter and not intra & inter


def test_short_cadence_average_is_available_at_interval_end(config):
    frame = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=7, freq="5min", tz="UTC"),
                          "power": [90, 3, 6, 9, 12, 15, np.nan]})
    clean_frame, stats = clean(frame, 10, config["data"], "UTC", native_minutes=5)
    assert list(clean_frame.index) == [pd.Timestamp("2021-01-01 00:15Z")]
    assert clean_frame.power.iloc[0] == 6
    assert stats["incomplete_resample_bins"] == 2


def test_short_budget_fallback_keeps_labels_disjoint(config):
    end = pd.Timestamp("2023-03-31", tz="UTC")
    test_start = end - pd.Timedelta(days=30)
    dev_start = test_start - pd.Timedelta(days=1)
    times = pd.date_range(end-pd.Timedelta(days=70), end, freq="15min", inclusive="left")
    times = times[~((times >= dev_start) & (times < dev_start+pd.Timedelta(hours=6)))]
    power = pd.Series(0.0, index=times)
    windows = window_power(power, 12, 1, 15)

    class Store:
        cfg = config

        def get(self, station):
            return power, windows, {"processed_sha256": "test"}

    episode = build_episode(Store(), "X", [], 1, (test_start, end))
    assert episode.manifest["development_allocation"] == "valid_windows_fallback"
    parts = list(episode.target_sets.values())
    assert min(map(len, parts)) >= 2
    assert all(a.last[-1] < b.first[0] for a, b in zip(parts, parts[1:]))
    assert parts[-1].first[0] == test_start
