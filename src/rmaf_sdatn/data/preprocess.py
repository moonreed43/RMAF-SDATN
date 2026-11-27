import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def read_table(path, **kwargs):
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, **kwargs)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, **kwargs)


def timestamps(values, timezone, date_format="mixed"):
    values = pd.Series(values, copy=False)
    offset = values.astype("string").str.contains(r"(?:Z|[+-]\d{2}:?\d{2}|\sUTC|\sGMT)$", case=False, na=False)
    if offset.any():
        without_offset = pd.to_datetime(values[~offset], format=date_format, errors="coerce")
        if without_offset.notna().any():
            raise ValueError("Timestamp input mixes timezone-aware and naive observations.")
        return pd.to_datetime(values, format=date_format, errors="coerce", utc=True)
    parsed = pd.to_datetime(values, format=date_format, errors="coerce")
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        if not pd.api.types.is_datetime64_any_dtype(parsed.dtype):
            aware = [value.tzinfo is not None for value in parsed if pd.notna(value)]
            if any(aware) and not all(aware):
                raise ValueError("Timestamp input mixes timezone-aware and naive observations.")
            return pd.to_datetime(values, format=date_format, errors="coerce", utc=True)
        parsed = parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    return parsed.dt.tz_convert("UTC")


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(frame, capacity, cfg, timezone, native_minutes=None, minimum_bin_coverage=1.0):
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError("A positive, independently documented capacity is required.")
    interval = cfg["interval_minutes"]
    native = interval if native_minutes is None else native_minutes
    if native <= 0 or interval % native:
        raise ValueError("Native sampling period must divide the prediction interval.")
    if not 0 < minimum_bin_coverage <= 1:
        raise ValueError("Minimum bin coverage must lie in (0,1].")
    times = timestamps(frame["timestamp"], timezone, cfg["date_format"])
    p = pd.to_numeric(frame["power"], errors="coerce").astype(float)
    invalid_time = times.isna()
    off_grid = times.notna() & (times != times.dt.floor(f"{native}min"))
    missing = ~np.isfinite(p)
    negative = p < 0
    above = p > capacity
    limit = cfg["max_normalized_power"]
    excessive = (p > capacity * limit) if limit is not None else np.zeros(len(p), dtype=bool)
    if cfg["negative"] not in {"drop", "clip", "keep"}:
        raise ValueError("Unknown negative power policy.")
    bad_negative = negative if cfg["negative"] == "drop" else np.zeros(len(p), dtype=bool)
    if cfg["negative"] == "clip":
        p = p.clip(lower=0)
    keep = ~(invalid_time | off_grid | missing | excessive | bad_negative)
    series = pd.Series(p[keep].to_numpy(), index=pd.DatetimeIndex(times[keep]), name="power").sort_index(kind="stable")
    duplicate = int(series.index.duplicated().sum())
    if cfg["duplicate"] == "mean":
        series = series.groupby(level=0).mean()
    elif cfg["duplicate"] in {"first", "last"}:
        series = series[~series.index.duplicated(keep=cfg["duplicate"])]
    else:
        raise ValueError("Unknown duplicate timestamp policy.")
    incomplete = 0
    if native != interval:
        grouped = series.resample(f"{interval}min", label="right", closed="right")
        good = grouped.count() >= np.ceil(interval / native * minimum_bin_coverage)
        incomplete = int((~good).sum())
        series = grouped.mean()[good]
    gaps = series.index.to_series().diff().dropna() / pd.Timedelta(minutes=interval)
    stats = {
        "raw_rows": len(frame), "invalid_timestamp": int(invalid_time.sum()),
        "off_grid": int(off_grid.sum()), "nonfinite_power": int(missing.sum()),
        "negative_power": int(negative.sum()), "negative_removed": int(np.sum(bad_negative & ~invalid_time & ~off_grid & ~missing & ~excessive)),
        "negative_clipped": int((negative & keep).sum()) if cfg["negative"] == "clip" else 0,
        "above_capacity": int(above.sum()), "above_configured_limit": int(np.sum(excessive)),
        "excluded_rows": int((~keep).sum()), "duplicate_rows_combined": duplicate,
        "incomplete_resample_bins": incomplete, "gap_count": int((gaps > 1).sum()),
        "missing_grid_slots": int((gaps - 1).clip(lower=0).sum()),
    }
    result = series.to_frame().assign(normalized_power=series / capacity)
    result.index.name = "timestamp"
    return result, stats


def _find_one(root, pattern):
    files = sorted(root.rglob(pattern))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected one {pattern!r} in {root}; found {len(files)}.")
    return files[0]


def _pvod_capacity(root, station):
    matches = list(root.rglob("metadata.csv"))
    if len(matches) != 1:
        return None
    metadata = pd.read_csv(matches[0], encoding="utf-8-sig")
    if "Station_ID" not in metadata and any("电站编号" in c for c in metadata):
        metadata = pd.read_csv(matches[0], header=1, encoding="utf-8-sig")
    id_column = next((c for c in metadata if c.strip().lower() == "station_id"), None)
    # The PVOD data descriptor documents Capacity in kW and power in MW.
    column = next((c for c in metadata if c.strip().lower() == "capacity" or
                   ("cap" in c.lower() and re.search(r"\bkw\b", c, re.I))), None)
    if id_column and column:
        selected = metadata[metadata[id_column].astype(str).str.lower() == f"station{station}"]
        if len(selected) == 1:
            return {"value": float(selected.iloc[0][column])/1000, "unit": "MW",
                    "source": f"{matches[0]}:{column}, kW converted to MW per PVOD descriptor"}
    return None


def station_input(station, spec, cfg):
    root = Path(cfg["root"])
    overrides = Path(cfg["capacity_overrides"])
    overrides = yaml.safe_load(overrides.read_text(encoding="utf-8")) if overrides.exists() else {}
    override = (overrides or {}).get(station, {})
    spec = {**spec, **override}
    dataset = spec["dataset"]
    files_root = root / "raw" / dataset
    native = spec.get("native_id", station)
    capacity = spec.get("capacity")
    if "files" in spec:
        paths = [Path(path) for path in spec["files"]]
    elif dataset == "pvod":
        paths = [_find_one(files_root, f"station{native}.csv")]
        capacity = capacity or _pvod_capacity(files_root, native)
    elif dataset == "state_grid":
        paths = sorted(files_root.rglob(f"Solar station site {native} (Nominal capacity-*.xlsx"))
        if len(paths) != 1:
            raise FileNotFoundError(f"Expected one original solar station {native} XLSX in {files_root}.")
        match = re.search(r"Nominal capacity-([\d.]+)\s*MW", paths[0].name, re.I)
        if match:
            capacity = capacity or {"value": float(match[1]), "unit": "MW", "source": paths[0].name}
    elif dataset == "pvdaq" and "pattern" in spec:
        paths = sorted(files_root.glob(spec["pattern"]))
        if not paths:
            raise FileNotFoundError(f"No PVDAQ power files match {spec['pattern']} in {files_root}.")
        metadata = files_root/f"{station}_system_metadata.json"
        if capacity is None and metadata.exists():
            payload = metadata.read_text(encoding="utf-8-sig")
            # Some OEDI JSON files have missing values in later instrument blocks.
            # Decode the independently valid System object and verify its ID.
            match = re.search(r'"System"\s*:\s*(\{)', payload)
            if match is None:
                raise ValueError(f"{metadata}: no System object.")
            record, _ = json.JSONDecoder().raw_decode(payload, match.start(1))
            if str(record["system_id"]) != station:
                raise ValueError(f"PVDAQ metadata ID differs from station {station}.")
            capacity = {"value": float(record["power(kW DC)"]), "unit": "kW",
                        "source": f"{metadata}:System.power(kW DC)"}
    else:
        raise FileNotFoundError(f"Specify verified PVDAQ power files and capacity for {station} in {cfg['capacity_overrides']}.")
    if not capacity or not capacity.get("source") or not capacity.get("unit"):
        raise ValueError(f"{station}: provide capacity.value, capacity.unit and capacity.source in {cfg['capacity_overrides']}.")
    input_timezone = spec.get("input_timezone", {"pvod": "UTC", "state_grid": "Asia/Shanghai"}.get(dataset, "UTC"))
    protocol_timezone = spec.get("timezone", {"pvod": "Asia/Shanghai", "state_grid": "Asia/Shanghai"}.get(dataset, "UTC"))
    frames = []
    for path in paths:
        options = dict(spec.get("read_options", {}))
        if dataset == "pvdaq" and path.suffix.lower() == ".csv" and spec.get("timestamp_column"):
            powers = spec.get("power_columns") or [spec.get("power_column")]
            options.setdefault("usecols", [spec["timestamp_column"], *powers])
        frame = read_table(path, **options)
        frame.columns = frame.columns.astype(str).str.strip()
        time_col = spec.get("timestamp_column")
        power_col = spec.get("power_column")
        if not time_col:
            time_col = next((name for name in ("date_time", "timestamp", "Time(year-month-day h:m:s)", "Time", "Date/time", "DateTime", "Datetime", "date/time", "Timestamp") if name in frame), None)
        if not power_col:
            power_col = next((name for name in ("power", "Power (MW)", "Power(MW)", "Power", "power (MW)", "Actual power (MW)") if name in frame), None)
        power_cols = spec.get("power_columns", [power_col])
        if time_col not in frame or any(column not in frame for column in power_cols):
            raise ValueError(f"{path}: set timestamp_column and power_column. Columns: {list(frame.columns)}")
        power = frame[power_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(power_cols))
        frames.append(pd.DataFrame({"timestamp": frame[time_col], "power": power * spec.get("power_multiplier", 1.0)}))
    return pd.concat(frames, ignore_index=True), paths, capacity, input_timezone, protocol_timezone, spec


def preprocess(station, cfg, model=None):
    frame, paths, cap, input_tz, tz, spec = station_input(station, cfg["stations"][station], cfg)
    capacity = float(cap["value"])
    clean_cfg = {**cfg, "negative": spec.get("negative", cfg["negative"])}
    result, stats = clean(frame, capacity, clean_cfg, input_tz, spec.get("native_minutes"), spec.get("minimum_bin_coverage", 1.0))
    if spec["dataset"] in cfg["periods"]:
        start, end = (pd.Timestamp(v, tz=tz) for v in cfg["periods"][spec["dataset"]])
    else:
        years = cfg["pvdaq"]["years"]
        start = pd.Timestamp(f"{min(years)-1}-12-01", tz=tz)
        end = pd.Timestamp(f"{max(years)+1}-01-01", tz=tz)
    raw_times = timestamps(frame.timestamp, input_tz, cfg["date_format"])
    raw_in_range = raw_times[(raw_times >= start) & (raw_times < end)]
    # Anchor China windows to recorded timestamps, even if the terminal power is invalid.
    grid = raw_in_range[raw_in_range == raw_in_range.dt.floor(f"{cfg['interval_minutes']}min")]
    stats["last_recorded_grid_timestamp"] = str(grid.max()) if len(grid) else None
    stats["clean_rows_outside_period"] = int(((result.index < start) | (result.index >= end)).sum())
    result = result[(result.index >= start) & (result.index < end)]
    if result.empty:
        from ..serialization import write_json
        failure = Path(cfg["root"])/"processed"/f"{station}.quality_failure.json"
        write_json({**stats, "station": station, "status": "no_valid_observations",
                    "capacity": capacity, "unit": cap["unit"], "capacity_source": cap["source"],
                    "period_start": str(start), "period_end": str(end), "files": [str(path) for path in paths]}, failure)
        raise ValueError(f"{station}: no clean observations in the configured period. Quality details: {failure}")
    folder = Path(cfg["root"]) / "processed"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{station}.csv"
    from ..serialization import atomic_destination
    with atomic_destination(destination) as temporary:
        result.to_csv(temporary)
    from .windows import window_power
    interval = cfg["interval_minutes"]
    window_counts = {}
    horizons = sorted({model["horizon"], *cfg.get("quality_horizons", [])}) if model else []
    for horizon in horizons:
        history = model["history"]
        windows = window_power(result.normalized_power, history, horizon, interval)
        possible = max(0, len(result)-history-horizon+1)
        window_counts[str(horizon)] = {"history": history, "horizon": horizon, "candidate_rows": possible,
                                       "valid": len(windows), "removed_nonconsecutive": possible-len(windows)}
    stats.update(station=station, dataset=spec["dataset"], capacity=capacity, unit=cap["unit"],
                 capacity_source=cap["source"], timezone=tz, input_timezone=input_tz,
                 negative_policy=clean_cfg["negative"],
                 native_interval_minutes=spec.get("native_minutes", cfg["interval_minutes"]),
                 minimum_bin_coverage=spec.get("minimum_bin_coverage", 1.0),
                 interval_minutes=cfg["interval_minutes"], first=str(result.index[0]),
                 last=str(result.index[-1]), retained=len(result), night_zero=int((result.power == 0).sum()),
                 files=[{"path": str(path), "sha256": sha256(path)} for path in paths],
                 preprocessing_version=2, processed_sha256=sha256(destination), window_quality=window_counts)
    from ..serialization import write_json
    write_json(stats, folder / f"{station}.json")
    # Successful reprocessing supersedes a previous empty-data quality report.
    failure = folder/f"{station}.quality_failure.json"
    if failure.exists():
        failure.unlink()
    return stats
