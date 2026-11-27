import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset

from .preprocess import sha256
from .windows import window_power

PARTS = ("reliability", "fusion", "adaptation", "validation")


@dataclass
class Episode:
    target: str
    sources: dict
    source_validation: dict
    target_sets: dict
    source_ramps: np.ndarray
    n_ref: int
    meta: dict
    manifest: dict

    def direct_labels(self):
        return ConcatDataset([self.target_sets[key].labeled() for key in PARTS[:-1]])


class StationData:
    def __init__(self, config):
        self.cfg = config
        self.folder = Path(config["data"]["root"]) / "processed"
        self.cache = {}
        self.train_cache = {}

    def get(self, station):
        if station not in self.cache:
            path = self.folder / f"{station}.csv"
            failure = self.folder/f"{station}.quality_failure.json"
            if failure.exists():
                raise ValueError(f"{station}: the latest preprocessing attempt failed. Repair the input and repeat preprocessing; see {failure}.")
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if meta["interval_minutes"] != self.cfg["data"]["interval_minutes"]:
                raise ValueError(f"{station}: processed interval differs from the experiment interval.")
            digest = sha256(path)
            if meta.get("processed_sha256", digest) != digest:
                raise ValueError(f"{station}: processed CSV differs from its quality record. Repeat preprocessing.")
            meta["processed_sha256"] = digest
            power = pd.read_csv(path, usecols=["timestamp", "power", "normalized_power"])
            capacity = meta.get("capacity", 0)
            if capacity <= 0 or not np.isfinite(capacity) or not np.allclose(
                    power.power.to_numpy()/capacity, power.normalized_power.to_numpy(), rtol=1e-6, atol=1e-8):
                raise ValueError(f"{station}: invalid capacity or normalized power differs from power/capacity.")
            index = pd.to_datetime(power.timestamp, utc=True)
            power = pd.Series(power.normalized_power.to_numpy(), index=pd.DatetimeIndex(index))
            model = self.cfg["model"]
            windows = window_power(power, model["history"], model["horizon"], meta["interval_minutes"])
            self.cache[station] = power, windows, meta
        return self.cache[station]


def source_ids(cfg, target, transfer):
    stations = cfg["data"]["stations"]
    region = stations[target]["region"]
    china = [key for key, spec in stations.items() if spec["region"] != "US" and key != target]
    if transfer == "external" and region == "US":
        return china
    if region == "US":
        raise ValueError("US targets require external transfer.")
    if transfer == "mixed":
        return china
    if transfer in {"intra", "inter"}:
        return [key for key in china if (stations[key]["region"] == region) == (transfer == "intra")]
    raise ValueError(f"Unknown transfer setting: {transfer}")


def budget_slots(days, ratios, interval, min_slots):
    return allocate(int(days * 1440 // interval), ratios, min_slots)


def allocate(total, ratios, min_slots):
    raw = total * np.asarray(ratios, dtype=float) / np.sum(ratios)
    if len(raw) != len(PARTS) or np.any(raw <= 0) or total < len(PARTS)*min_slots:
        raise ValueError(f"{total} eligible positions cannot provide {min_slots} per target subset.")
    counts = np.maximum(np.floor(raw).astype(int), min_slots)
    while counts.sum() > total:
        candidates = np.where(counts > min_slots, counts - raw, -np.inf)
        counts[np.argmax(candidates)] -= 1
    while counts.sum() < total:
        counts[np.argmax(raw - counts)] += 1
    return counts


def test_windows(store, target, failures=None):
    """Return eligible episodes; optionally report unavailable seasons separately.

    A missing season must not hide clean independent seasons of the same system.
    The strict default still raises when any requested seasonal test is missing.
    """
    d = store.cfg["data"]
    power, windows, meta = store.get(target)
    step = pd.Timedelta(minutes=d["interval_minutes"])
    if meta["dataset"] != "pvdaq":
        last = meta.get("last_recorded_grid_timestamp")
        end = (pd.Timestamp(last) if last else store.get(target)[0].index[-1]) + step
        return {"final": (end - pd.Timedelta(days=d["test_days"]), end)}
    timezone = meta["timezone"]
    pv = d["pvdaq"]
    recorded_last = meta.get("last_recorded_grid_timestamp")
    available_end = (pd.Timestamp(recorded_last) if recorded_last else power.index[-1]) + step
    desired = {f"{y}-{s}": pd.Timestamp(f"{y}-{s}", tz=timezone).tz_convert("UTC")
               for y in pv["years"] for s in pv["starts"]}
    days = pd.Timedelta(days=d["test_days"])
    required = pv["minimum_coverage"] * (days // step - store.cfg["model"]["horizon"] + 1)
    reserved = [(s, s + days) for label, s in desired.items() if not label.endswith("12-01")]
    episodes = {}
    for label, start in desired.items():
        end = start + days
        if label.endswith("12-01") and (end > available_end or len(windows.between(start, end)) < required):
            minimum = pd.Timestamp(f"{start.year}-01-01", tz=timezone).tz_convert("UTC")
            # Find the latest eligible start on the 15-minute grid.
            first_ns = windows.first.to_numpy(dtype="datetime64[ns]")
            last_ns = windows.last.to_numpy(dtype="datetime64[ns]")
            while start >= minimum:
                end = start + days
                left = np.searchsorted(first_ns, start.to_datetime64(), side="left")
                right = np.searchsorted(last_ns, end.to_datetime64(), side="left")
                if end <= available_end and not any(start < b and end > a for a, b in reserved) and right - left >= required:
                    break
                start -= step
            else:
                error = ValueError(f"{target} {label}: no nonoverlapping window meets coverage.")
                if failures is None:
                    raise error
                failures[label] = str(error)
                continue
        if end > available_end or len(windows.between(start, end)) < required:
            error = ValueError(f"{target} {label}: insufficient clean test windows.")
            if failures is None:
                raise error
            failures[label] = str(error)
            continue
        episodes[label] = (start, end)
        if label.endswith("12-01"):
            reserved.append((start, end))
    # A seasonal development/reference block cannot overlap any reserved test
    # period. Move the whole block earlier when a December fallback requires it.
    development = pd.Timedelta(days=max(d["reference_days"], max(store.cfg["experiment"]["budgets"])))
    history_margin = step * store.cfg["model"]["history"] if d["allow_prior_history"] else pd.Timedelta(0)
    windows = list(episodes.values())
    result = {}
    for name, (start, end) in episodes.items():
        dev_end = start
        while overlaps := [a for a, b in windows if a < dev_end and b > dev_end-development-history_margin]:
            dev_end = min(overlaps)
        result[name] = (start, end, dev_end)
    return result


def build_episode(store, target, sources, budget, window):
    cfg = store.cfg
    data = cfg["data"]
    interval = pd.Timedelta(minutes=data["interval_minutes"])
    _, win, meta = store.get(target)
    start_test, end_test = window[:2]
    dev_end = window[2] if len(window) == 3 else start_test
    if dev_end > start_test:
        raise ValueError("Target development block must precede the test window.")
    allow_history = data["allow_prior_history"]
    min_slots = data["min_windows"] + cfg["model"]["horizon"] - 1
    if not allow_history:
        min_slots += cfg["model"]["history"]
    slots = budget_slots(budget, data["ratios"], data["interval_minutes"], min_slots)
    dev_start = dev_end - int(slots.sum()) * interval
    boundaries = [dev_start, *(dev_start + int(count)*interval for count in np.cumsum(slots))]
    provisional = [win.between(a, b, allow_history) for a,b in zip(boundaries, boundaries[1:])]
    allocation_by_windows = False
    if (data.get("short_budget_valid_fallback", True) and budget < data["reference_days"]
            and min(map(len, provisional)) < data["min_windows"]):
        valid = win.between(dev_start, dev_end, allow_history)
        parts_by_count = allocate(len(valid), data["ratios"], min_slots)
        cuts = [valid.first[int(i)] for i in np.cumsum(parts_by_count)[:-1]]
        boundaries = [dev_start, *cuts, dev_end]
        allocation_by_windows = True
    parts, info = {}, {}
    ranges = [*zip(boundaries, boundaries[1:]), (start_test, end_test)]
    for name, (left, right) in zip(PARTS + ("test",), ranges):
        parts[name] = win.between(left, right, allow_history)
        info[name] = {"start": str(left), "end_exclusive": str(right), "windows": len(parts[name]),
                      "first_label": str(parts[name].first[0]) if len(parts[name]) else None,
                      "last_label": str(parts[name].last[-1]) if len(parts[name]) else None}
        if len(parts[name]) < data["min_windows"]:
            raise ValueError(f"{target} {name}: {len(parts[name])} valid windows in [{left}, {right}).")
    reference_start = dev_end - pd.Timedelta(days=data["reference_days"])
    n_ref = len(win.between(reference_start, dev_end, allow_history))
    if not n_ref:
        raise ValueError("No valid reference samples.")
    source_train, source_val, ramps, source_info = {}, {}, [], {}
    if target in sources or len(set(sources)) != len(sources):
        raise ValueError("Sources must be unique non-target stations.")
    for source in sources:
        if source not in store.train_cache:
            power, source_win, source_meta = store.get(source)
            frac = cfg["training"]["source_validation_fraction"]
            if not 0 < frac < 1:
                raise ValueError("source_validation_fraction must be in (0,1).")
            cutoff = power.index[int(len(power) * (1-frac))]
            train_win = source_win.between(power.index[0], cutoff)
            val_win = source_win.between(cutoff, power.index[-1] + interval)
            if min(len(train_win), len(val_win)) < data["min_windows"]:
                raise ValueError(f"{source}: insufficient valid source windows.")
            p = power.loc[power.index < cutoff]
            consecutive = p.index.to_series().diff() == interval
            ramp = p.diff().abs()[consecutive].to_numpy()
            source_manifest = {"train": len(train_win), "validation": len(val_win), "cutoff": str(cutoff),
                               "processed_sha256": source_meta["processed_sha256"]}
            if source_meta.get("preprocessing_version", 1) >= 2:
                source_manifest["metadata_sha256"] = sha256(store.folder/f"{source}.json")
            store.train_cache[source] = train_win, val_win, ramp, source_manifest
        source_train[source], source_val[source], ramp, source_info[source] = store.train_cache[source]
        ramps.append(ramp)
    manifest = {"target": target, "budget_days": budget, "source_sets": source_info, "target_sets": info,
                "n_ref": n_ref, "reference_start": str(reference_start),
                "target_sha256": meta["processed_sha256"], "allow_prior_history": allow_history}
    if allocation_by_windows:
        manifest["development_allocation"] = "valid_windows_fallback"
    if meta.get("preprocessing_version", 1) >= 2:
        manifest["target_metadata_sha256"] = sha256(store.folder/f"{target}.json")
    return Episode(target, source_train, source_val, parts, np.concatenate(ramps) if ramps else np.empty(0), n_ref, meta, manifest)
