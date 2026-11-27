import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import merge
from .data.splits import StationData, build_episode, source_ids, test_windows
from .serialization import write_json, write_csv
from .locking import file_lock
from .training.runner import run_trial


def record_failure(root, context, error):
    with file_lock(Path(root)/"results.lock", timeout=300):
        path = root/"failures.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        remaining = [row for row in previous if row["context"] != context]
        remaining.append({"context": context, "error_type": type(error).__name__, "reason": str(error),
                          "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(remaining, path)


def resolve_failure(root, context):
    with file_lock(Path(root)/"results.lock", timeout=300):
        path = root/"failures.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            remaining = [row for row in previous if row["context"] != context]
            if len(remaining) != len(previous):
                write_json(remaining, path)


def selection_jobs(config, target, pool, store, budget, times, seed, prefix):
    analysis = config["experiment"]["analysis"]
    if analysis == "standard":
        return [("all", pool, [])], {}
    if analysis == "reliability":
        return ([("all", pool, [])] + [(f"only-{s}", [s], []) for s in pool]
                + [(f"without-{s}", [x for x in pool if x != s], []) for s in pool if len(pool) > 1]), {}
    if analysis != "stability" or len(pool) != 7:
        raise ValueError("Stability requires the seven main-fold source stations.")
    reference = build_episode(store, target, pool, budget, times)
    rank_root = prefix/"validation_ranking"
    target_only = run_trial(config, reference, seed, rank_root/"target_only", "target_only",
                            validation_only=True)["validation_rmse"]
    gains = {}
    for source in pool:
        episode = build_episode(store, target, [source], budget, times)
        score = run_trial(config, episode, seed, rank_root/source, "multi_source", validation_only=True)["validation_rmse"]
        gains[source] = 1 - score / max(target_only, config["algorithm"]["epsilon"])
    order = sorted(pool, key=lambda name: (-gains[name], name))
    incompatible = set(order[4:])
    jobs = [("four-"+"-".join(combo), list(combo), sorted(incompatible.intersection(combo)))
            for combo in itertools.combinations(pool, 4)]
    write_json({"validation_gain": gains, "compatible": order[:4], "incompatible": order[4:]},
               prefix/"source_ranking.json")
    return jobs, gains


def experiments(config, target_filter=None, methods_filter=None, variant_filter=None, source_override=None,
                budget_filter=None, transfer_filter=None, seed_filter=None):
    root = Path(config["experiment"]["output"])/config["experiment"]["name"]
    try:
        return _execute(config, target_filter, methods_filter, variant_filter, source_override,
                        budget_filter, transfer_filter, seed_filter)
    except BaseException:
        # Completed trials remain reviewable after a later failure or interruption.
        try:
            recover_results(root)
        except Exception as recovery_error:
            import logging
            logging.getLogger(__name__).error("Could not rebuild the result index: %s", recovery_error)
        raise


def _execute(config, target_filter=None, methods_filter=None, variant_filter=None, source_override=None,
             budget_filter=None, transfer_filter=None, seed_filter=None):
    exp = config["experiment"]
    root = Path(exp["output"])/exp["name"]
    root.mkdir(parents=True, exist_ok=True)
    store = StationData(config)
    rows, comparisons = [], []
    methods = methods_filter or exp["methods"]
    seeds = seed_filter or config["training"]["seeds"]
    budgets = budget_filter or exp["budgets"]
    transfers = transfer_filter or exp["transfers"]
    variants = [name for name in exp["variants"] if not variant_filter or name in variant_filter]
    from .validation import SUPPORTED_METHODS
    if not set(methods) <= SUPPORTED_METHODS:
        raise ValueError(f"Unknown methods: {set(methods) - SUPPORTED_METHODS}")
    if variant_filter and not set(variant_filter) <= set(exp["variants"]):
        raise ValueError(f"Unknown variants: {set(variant_filter) - set(exp['variants'])}")
    if not variants:
        raise ValueError("No matching experiment variants.")
    for target in target_filter or exp["targets"]:
        try:
            window_failures = {}
            seasons = test_windows(store, target, failures=window_failures) if exp.get("keep_going", False) else test_windows(store, target)
        except (OSError, ValueError, KeyError) as error:
            record_failure(root, {"target": target}, error)
            if not exp.get("keep_going", False):
                raise
            continue
        resolve_failure(root, {"target": target})
        for label, reason in window_failures.items():
            record_failure(root, {"target": target, "window": label}, ValueError(reason))
        for label, times in seasons.items():
            resolve_failure(root, {"target": target, "window": label})
            for transfer, budget, seed in itertools.product(transfers, budgets, seeds):
                prefix = root/target/label/transfer/f"budget-{budget}"/f"seed-{seed}"
                selection_context = {"target": target, "window": label, "transfer": transfer,
                                     "budget": budget, "seed": seed, "phase": "source_selection"}
                try:
                    pool = source_override or source_ids(config, target, transfer)
                    jobs, gains = selection_jobs(config, target, pool, store, budget, times, seed, prefix)
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    record_failure(root, selection_context, error)
                    if not exp.get("keep_going", False):
                        raise
                    continue
                resolve_failure(root, selection_context)
                method_results = {}
                for source_case, sources, incompatible in jobs:
                    for method in methods:
                        if method == "target_only" and source_case != "all" and ("baseline", method, "full") in method_results:
                            continue
                        modes = variants if method == "rmaf" else ["full"]
                        for variant in modes:
                            if source_case != "all" and exp["analysis"] == "reliability" and variant != "full":
                                continue
                            cfg = merge(config, exp["variants"].get(variant, {}) if method == "rmaf" else {})
                            same_data = (cfg["data"] == config["data"]
                                         and cfg["model"]["history"] == config["model"]["history"]
                                         and cfg["model"]["horizon"] == config["model"]["horizon"]
                                         and cfg["training"]["source_validation_fraction"] == config["training"]["source_validation_fraction"])
                            variant_store = store if same_data else StationData(cfg)
                            directory = prefix/source_case/f"{method}-{variant}"
                            context = {"target": target, "window": label, "transfer": transfer,
                                       "budget": budget, "seed": seed, "source_case": source_case,
                                       "method": method, "variant": variant}
                            try:
                                episode = build_episode(variant_store, target, sources, budget, times)
                                result = run_trial(cfg, episode, seed, directory, method, until=exp.get("evaluate_stage", 4),
                                                   score=True, stage1_cache=root/"source_checkpoints")
                            except (OSError, ValueError, KeyError, RuntimeError) as error:
                                record_failure(root, context, error)
                                if not exp.get("keep_going", False):
                                    raise
                                continue
                            resolve_failure(root, context)
                            method_results[(source_case, method, variant)] = result
                            if method == "target_only":
                                method_results[("baseline", method, "full")] = result
                            record = {key: value for key, value in result.items() if key != "metrics"}
                            record.update(window=label, transfer=transfer, budget=budget, variant=variant,
                                          source_case=source_case, sources=" ".join(sources),
                                          incompatible_count=len(incompatible) if exp["analysis"] == "stability" else np.nan,
                                          path=str(directory))
                            write_json({"record": record, "metrics": result["metrics"]}, directory/"experiment_record.json")
                            for metric in result["metrics"]:
                                rows.append({**record, **metric})
                            if exp["analysis"] == "stability" and method == "rmaf" and (directory/"reliability.csv").exists():
                                reliability = pd.read_csv(directory/"reliability.csv")
                                fused = reliability.loc[reliability.source.isin(incompatible), "weight"].sum()
                                mass_path = directory/"alignment_mass.json"
                                mass = json.loads(mass_path.read_text(encoding="utf-8")) if mass_path.exists() else {}
                                denominator = sum(mass.values())
                                alignment = sum(mass.get(s, 0) for s in incompatible)/denominator if denominator else np.nan
                                comparisons.append({"target": target, "window": label, "transfer": transfer,
                                                    "budget": budget, "seed": seed, "variant": variant,
                                                    "source_case": source_case, "incompatible_count": len(incompatible),
                                                    "incompatible_fusion_weight": fused,
                                                    "incompatible_alignment_share": alignment})
                if exp["analysis"] == "reliability" and ("all", "rmaf", "full") in method_results:
                    rel = pd.read_csv(prefix/"all/rmaf-full/reliability.csv")
                    base = method_results.get(("baseline", "target_only", "full"))
                    base_rmse = _rmse(base) if base else np.nan
                    full_rmse = _rmse(method_results[("all", "rmaf", "full")])
                    for row in rel.to_dict("records"):
                        source = row["source"]
                        single = method_results.get((f"only-{source}", "rmaf", "full"))
                        omitted = method_results.get((f"without-{source}", "rmaf", "full"))
                        comparisons.append({"target": target, "window": label, "transfer": transfer,
                                            "budget": budget, "seed": seed, **row,
                                            "single_source_rmse": _rmse(single) if single else np.nan,
                                            "single_source_gain": 1-_rmse(single)/base_rmse if single and base_rmse else np.nan,
                                            "leave_one_out_rmse_change": _rmse(omitted)-full_rmse if omitted else np.nan})
                if gains:
                    for source, gain in gains.items():
                        comparisons.append({"target": target, "window": label, "transfer": transfer,
                                            "budget": budget, "seed": seed, "source": source,
                                            "validation_single_source_gain": gain})
    if not rows:
        if exp.get("keep_going", False):
            recover_results(root)
            return root
        raise ValueError("No trials were evaluated.")
    dataframe = publish_results(pd.DataFrame(rows), root)
    if comparisons:
        frames = pd.DataFrame(comparisons)
        previous_analysis = root/"source_analysis.csv"
        if previous_analysis.exists():
            frames = pd.concat([pd.read_csv(previous_analysis, dtype={"target": str}), frames], ignore_index=True)
        dimensions = [c for c in ("target", "window", "transfer", "budget", "seed", "variant", "source_case", "source") if c in frames]
        frames = frames.drop_duplicates(dimensions, keep="last")
        write_csv(frames, previous_analysis)
        if exp["analysis"] == "reliability" and "error" in frames:
            frames["negative_error"] = -frames["error"]
            frames["negative_delta"] = -frames["delta"]
            correlations = frames.groupby([c for c in ("target", "window", "transfer", "budget", "seed") if c in frames])[
                ["rho", "negative_error", "negative_delta", "weight", "single_source_gain", "leave_one_out_rmse_change"]
            ].corr(method="spearman")
            write_csv(correlations, root/"source_spearman.csv", index=True)
    from .evaluation.visualization import plot_experiment
    plot_experiment(dataframe, config, root/"figures")
    return root


def recover_results(root):
    root = Path(root)
    rows = []
    if root.exists():
        for path in sorted(root.rglob("experiment_record.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            rows.extend({**record["record"], **metric} for metric in record["metrics"])
    if not rows and not (root/"runs.csv").exists():
        return None
    return publish_results(pd.DataFrame(rows), root)


def publish_results(dataframe, root):
    root = Path(root)
    with file_lock(root/"results.lock", timeout=300):
        return _publish_results(dataframe, root)


def _publish_results(dataframe, root):
    index_columns = ["path", "step", "subset"]
    previous = root/"runs.csv"
    if previous.exists():
        existing = pd.read_csv(previous, dtype={"target": str})
        existing = existing.drop(columns=["negative_transfer", "transfer_gain", "target_only_rmse"], errors="ignore")
        dataframe = pd.concat([existing, dataframe], ignore_index=True).drop_duplicates(index_columns, keep="last")
    keys = ["target", "window", "transfer", "budget", "seed", "step", "subset"]
    comparator = dataframe[dataframe.method == "target_only"].drop_duplicates(keys)
    dataframe = dataframe.merge(comparator[keys+["rmse"]].rename(columns={"rmse": "target_only_rmse"}), on=keys, how="left")
    is_transfer = ~dataframe.method.isin(["target_only", "persistence"])
    paired = dataframe.target_only_rmse.notna() & dataframe.rmse.notna() & is_transfer & (dataframe.subset == "all")
    dataframe["negative_transfer"] = np.where(paired, dataframe.rmse > dataframe.target_only_rmse, np.nan)
    dataframe["transfer_gain"] = np.where(paired, 1-dataframe.rmse/dataframe.target_only_rmse.replace(0, np.nan), np.nan)
    write_csv(dataframe, root/"runs.csv")
    summarize(dataframe, root)
    return dataframe


def _rmse(result):
    return next(row["rmse"] for row in result["metrics"] if row["subset"] == "all")


def summarize(df, root):
    groups = ["target", "window", "transfer", "budget", "method", "variant", "source_case", "step", "subset"]
    keys = ["rmse", "mae", "r2", "training_seconds", "single_sample_latency_ms", "negative_transfer", "transfer_gain"]
    summary = df.groupby(groups, dropna=False)[keys].agg(["mean", "std", "count"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.reset_index().to_csv(root/"mean_sd.csv", index=False)
    main = df[df.subset == "all"].copy()
    stat_keys = ["window", "transfer", "budget", "method", "variant", "step"]
    main.groupby(stat_keys, dropna=False).agg(
        ntr=("negative_transfer", "mean"), negative_count=("negative_transfer", "sum"),
        paired_count=("negative_transfer", "count"), mean_gain=("transfer_gain", "mean"),
        worst_gain=("transfer_gain", "min")).to_csv(root/"negative_transfer.csv")
    chosen = main[main.source_case == "all"]
    if not chosen.empty:
        fold = chosen.groupby(["target", "window", "transfer", "budget", "method", "variant", "step"]).rmse.mean().reset_index()
        fold["year"] = np.where(fold.transfer == "external", fold.window.str[:4], "all")
        by_site_year = fold.groupby(["target", "year", "transfer", "budget", "method", "variant", "step"]).rmse.mean().reset_index()
        by_site_year.to_csv(root/"target_year_rmse.csv", index=False)
        dims = ["transfer", "budget", "method", "variant", "step"]
        by_site_year.groupby(dims).rmse.agg(["mean", "std", "count"]).to_csv(root/"across_targets.csv")
        if (fold.transfer == "external").any():
            by_site_year.groupby(dims+["year"]).rmse.agg(["mean", "std", "count"]).to_csv(root/"annual_rmse.csv")
            by_site_year.groupby(dims+["target"]).rmse.agg(["mean", "std", "count"]).to_csv(root/"external_system_rmse.csv")
    stability = main[main.incompatible_count.notna()]
    if not stability.empty:
        group = ["method", "variant", "transfer", "budget", "step"]
        grouped = stability.groupby(group)
        table = grouped.agg(rmse=("rmse", "mean"), mean_gain=("transfer_gain", "mean"),
                            worst_gain=("transfer_gain", "min"), ntr=("negative_transfer", "mean"),
                            negative_count=("negative_transfer", "sum"), paired_count=("negative_transfer", "count"))
        across_sources = stability.groupby(group+["target", "window", "seed"]).rmse.std()
        table["source_set_sd"] = across_sources.groupby(level=group).mean()
        table.to_csv(root/"stability.csv")
        contamination = stability.groupby(group+["incompatible_count"]).rmse.agg(["mean", "std", "count"])
        endpoint = contamination["mean"].unstack("incompatible_count")
        if 0 in endpoint and 3 in endpoint:
            relative = endpoint[3]/endpoint[0].replace(0, np.nan)-1
            relative.rename("relative_rmse_degradation_0_to_75pct").to_csv(root/"contamination_degradation.csv")
        contamination.to_csv(root/"contamination.csv")
