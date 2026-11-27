"""Fail early on unsupported experiment settings and contradictory protocols."""
import math

SUPPORTED_METHODS = {"rmaf", "persistence", "target_only", "multi_source", "dann", "mmd", "coral",
                     "patchtst", "timesnet", "cyclenet"}


def validate_config(config):
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    def positive(value):
        return isinstance(value, (int, float)) and math.isfinite(value) and value > 0

    def integer(value):
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    def choice(section, name, values):
        require(section.get(name) in values, f"{name} must be one of {sorted(values)}")

    m, d, t, a, exp, e = (config[k] for k in ("model", "data", "training", "algorithm", "experiment", "evaluation"))
    for key in ("history", "horizon", "hidden", "layers", "rank", "embedding", "shared_dim", "supplementary_dim", "discriminator_hidden"):
        require(integer(m.get(key)), f"model.{key} must be a positive integer")
    require(0 <= m["dropout"] < 1, "model.dropout must lie in [0,1)")
    require(positive(m["alpha"]), "model.alpha must be positive")
    require(bool(m["scales"]) and all(integer(s) and s < m["history"] for s in m["scales"]),
            "Evolution scales must be positive integer lags smaller than history")
    require(len(set(m["scales"])) == len(m["scales"]), "Evolution lags must be unique")
    require(integer(d["interval_minutes"]) and 1440 % d["interval_minutes"] == 0,
            "data.interval_minutes must divide a day")
    require(len(d["ratios"]) == 4 and all(positive(x) for x in d["ratios"]), "data.ratios requires four positive values")
    for key in ("min_windows", "test_days", "reference_days"):
        require(integer(d[key]), f"data.{key} must be positive")
    choice(d, "negative", {"drop", "clip", "keep"})
    choice(d, "duplicate", {"mean", "first", "last"})
    require(0 < d["pvdaq"]["minimum_coverage"] <= 1, "PVDAQ minimum coverage must lie in (0,1]")
    for key in ("batch_size", "patience", "cpu_threads"):
        require(integer(t[key]), f"training.{key} must be a positive integer")
    require(isinstance(t["workers"], int) and t["workers"] >= 0, "training.workers must be nonnegative")
    require(t["source_batches_per_epoch"] is None or integer(t["source_batches_per_epoch"]),
            "source_batches_per_epoch must be null or a positive integer")
    require(0 < t["source_validation_fraction"] < 1, "source_validation_fraction must lie in (0,1)")
    require(positive(t["grad_clip"]) and t["weight_decay"] >= 0, "Invalid clipping norm or weight decay")
    require(bool(t["seeds"]) and all(isinstance(s, int) and 0 <= s < 2**32 for s in t["seeds"]),
            "training.seeds must contain integers in [0,2**32)")
    require(all(integer(x) for x in t["epochs"].values()), "Each training stage requires at least one epoch")
    require(positive(a["epsilon"]) and a["adapter_penalty"] >= 0, "Invalid epsilon or adapter regularization")
    r, f, align, anchor = (a[k] for k in ("reliability", "fusion", "alignment", "anchoring"))
    require(0 <= r["eta"] <= 1 and positive(r["temperature"]) and r["covariance"] >= 0, "Invalid reliability parameters")
    choice(f, "mode", {"constrained", "uniform", "prior_only", "objective_only", "no_prior_penalty"})
    choice(f, "mutation", {"gaussian", "uniform"})
    require(integer(f["population"]) and f["population"] >= 2 and integer(f["generations"]), "Invalid fusion search budget")
    require(0 < f["elite_ratio"] < 1 and f["mutation_std"] >= 0 and f["init_std"] >= 0 and f["prior_penalty"] >= 0,
            "Invalid fusion mutation, elite ratio or prior penalty")
    choice(align, "branch", {"shared", "both"})
    choice(align, "discriminator", {"per_source", "global"})
    choice(align, "station_weights", {"reliability", "uniform"})
    choice(align, "gate", {"online", "none", "high", "low"})
    require(len(align["gate_range"]) == 2 and 0 <= align["gate_range"][0] <= align["gate_range"][1],
            "gate_range must have ordered nonnegative bounds")
    require(0 < align["gate_fraction"] <= 1 and positive(align["gamma"]) and align["lambda_max"] >= 0, "Invalid alignment schedule")
    choice(anchor, "mode", {"none", "fixed", "scarcity"})
    require(all(anchor[k] >= 0 for k in ("backbone", "adapter", "evolution", "multiplier")), "Negative anchoring strength")
    require(positive(t["lr"]["initial"]), "Initial learning rate must be positive")
    for stage in ("adversarial", "target"):
        require(all(positive(lr) for lr in t["lr"][stage].values()), f"Invalid {stage} learning rate")
    lr = t["lr"]["target"]
    require(lr["backbone"] < min(lr["adapter"], lr["evolution"]) <= max(lr["adapter"], lr["evolution"]) < lr["head"],
            "Target learning rates require backbone < adapter/evolution < head")
    choice(exp, "analysis", {"standard", "reliability", "stability"})
    require(bool(exp["targets"]) and all(str(s) in d["stations"] for s in exp["targets"]), "Unregistered experiment target")
    require(bool(exp["budgets"]) and all(integer(b) for b in exp["budgets"]), "Invalid target budgets")
    require(set(exp["methods"]) <= SUPPORTED_METHODS and bool(exp["methods"]), "Unknown or empty experiment methods")
    require(set(exp["transfers"]) <= {"mixed", "intra", "inter", "external"} and bool(exp["transfers"]), "Unknown transfer mode")
    require(exp.get("evaluate_stage", 4) in {2, 3, 4}, "evaluate_stage must be 2, 3 or 4")
    require(isinstance(exp["variants"], dict) and bool(exp["variants"]), "At least one named variant is required")
    require(integer(e["diagnostic_samples"]) and integer(e["latency_repeats"]), "Invalid diagnostic or latency sample count")
    require(0 <= e["ramp_quantile"] <= 1, "ramp_quantile must lie in [0,1]")
    require(all(integer(s) and s <= m["horizon"] for s in e.get("steps", [1])), "Evaluation step exceeds the output horizon")
    for method in set(exp["methods"]) & {"cyclenet", "patchtst", "timesnet"}:
        settings = config["baselines"][method]
        if method == "cyclenet":
            require(integer(settings["period"]) and settings["interval_minutes"] == d["interval_minutes"],
                    "CycleNet requires a positive period and matching data interval")
            choice(settings, "head", {"linear", "mlp"})
        if method == "patchtst":
            require(0 < settings["patch_length"] <= m["history"] and integer(settings["patch_stride"]),
                    "PatchTST patch length must fit the input window")
            require(settings["width"] % settings["heads"] == 0, "PatchTST width must divide evenly into heads")
        if method == "timesnet":
            require(0 < settings["periods"] <= (m["history"] + m["horizon"])//2,
                    "TimesNet top-k exceeds available nonzero frequencies")
    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))
    return config
