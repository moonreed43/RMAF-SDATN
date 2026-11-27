import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
import yaml

from ..evaluation.baselines import run_baseline
from ..config import load_config
from ..evaluation.complexity import complexity, sync
from ..evaluation.diagnostics import diagnostics
from ..evaluation.metrics import predict_batches, score_predictions
from ..inference import save_predictor
from ..models.cyclenet import supervised_dataset
from ..locking import file_lock
from ..serialization import atomic_save, load_weights, write_json
from .common import Training, configure
from .stage1 import run_stage1
from .stage2 import run_stage2
from .stage3 import from_stage, run_stage3
from .stage4 import run_stage4
from ..validation import validate_config
from .. import __version__

STAGES = (run_stage1, run_stage2, run_stage3, run_stage4)


def _signature(config, manifest, seed, method):
    text = json.dumps([config, manifest, seed, method], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _run_settings(config, method):
    baseline = config["baselines"].get(method, {}) if method in {"cyclenet", "patchtst", "timesnet"} else {}
    return {"model": config["model"], "algorithm": config["algorithm"], "baselines": baseline,
            "training": {key: value for key, value in config["training"].items() if key != "seeds"}}


def _source_signature(config, episode, seed):
    training = config["training"]
    settings = {key: training[key] for key in ("batch_size", "weight_decay", "grad_clip", "patience",
                                                "workers", "deterministic", "device", "cpu_threads",
                                                "source_batches_per_epoch")}
    settings["epochs"] = {name: training["epochs"][name] for name in ("shared", "adapter")}
    settings["lr"] = training["lr"]["initial"]
    algorithm = {key: config["algorithm"][key] for key in ("adapter_penalty", "independent_full_models")}
    sources = list(episode.manifest["source_sets"].items())
    return _signature([config["model"], settings, algorithm], sources, seed, "stage1-v1")


def _stage_checkpoint(payload, path, tr, signature, seconds, peak):
    atomic_save({"signature": signature, "payload": payload, "seconds": seconds, "peak": peak,
                 "rng": tr.rng_state(),
                 "torch_rng": torch.get_rng_state(), "loader_rng": tr.generator.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if tr.device.type == "cuda" else []}, path)


def _restore(package, tr, signature):
    if package["signature"] != signature:
        raise ValueError("Checkpoint data, seed or configuration differs. Use a new run directory.")
    if "rng" in package:
        tr.restore_rng(package["rng"])
        return package["payload"]
    torch.set_rng_state(package["torch_rng"])
    tr.generator.set_state(package["loader_rng"])
    if tr.device.type == "cuda" and package["cuda_rng"]:
        torch.cuda.set_rng_state_all(package["cuda_rng"])
    return package["payload"]


def run_trial(config, episode, seed, directory, method="rmaf", until=4, score=False, validation_only=False,
              stage1_cache=None):
    with file_lock(Path(directory)/"run.lock"):
        return _run_trial(config, episode, seed, directory, method, until, score, validation_only, stage1_cache)


def _run_trial(config, episode, seed, directory, method, until, score, validation_only, stage1_cache):
    invocation_started = perf_counter()
    validate_config(config)
    if method != "persistence" and method != "target_only" and not episode.sources:
        raise ValueError(f"{method} requires at least one non-target source station.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    signature = _signature(_run_settings(config, method), episode.manifest, seed, method)
    identity_file = directory/"identity.json"
    if identity_file.exists():
        existing = json.loads(identity_file.read_text(encoding="utf-8"))
        if existing["signature"] != signature:
            # Preserve earlier checkpoints when edits affect other stations,
            # evaluation output, or which runs the batch command dispatches.
            old_config = load_config(directory/"config.yaml")
            old_manifest = json.loads((directory/"split.json").read_text(encoding="utf-8"))
            if (existing["seed"] != seed or existing["method"] != method or old_manifest != episode.manifest
                    or _run_settings(old_config, method) != _run_settings(config, method)):
                raise ValueError(f"Existing run at {directory} has different inputs or training settings.")
            signature = existing["signature"]
    write_json({"signature": signature, "seed": seed, "method": method}, identity_file)
    write_json(episode.manifest, directory/"split.json")
    (directory/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    device = configure(seed, config["training"])
    environment_path = directory/"environment.json"
    if not environment_path.exists():
        existing_checkpoints = any(directory.glob("stage*.pt")) or (directory/"baseline.pt").exists()
        write_json({"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "package_version": __version__, "python": platform.python_version(),
                    "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(), "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "seed": seed, "deterministic": config["training"]["deterministic"],
                    "scope": "restored_run_inspection" if existing_checkpoints else "training"}, environment_path)
    tr = Training(config, seed, device, directory/"progress")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak = None
    training_seconds = 0.0
    executed_training_seconds = 0.0
    states = {}
    reused_source_stage = False
    if method == "rmaf":
        previous = None
        for stage, run in enumerate(STAGES[:until], start=1):
            path = directory/f"stage{stage}.pt"
            source_cache = None
            if stage == 1 and stage1_cache is not None:
                source_signature = _source_signature(config, episode, seed)
                source_cache = stage1_cache/f"{source_signature}.pt"
            if path.exists():
                package = load_weights(path)
                previous = _restore(package, tr, signature)
                seconds, stage_peak = package["seconds"], package["peak"]
                reused_source_stage = reused_source_stage or package.get("source_cache_hit", False)
            elif source_cache and source_cache.exists():
                package = load_weights(source_cache)
                previous = _restore(package, tr, source_signature)
                seconds, stage_peak = package["seconds"], package["peak"]
                atomic_save({**package, "signature": signature, "source_cache_hit": True}, path)
                reused_source_stage = True
            else:
                print(f"{directory.name} | stage {stage}", flush=True)
                sync(device)
                start = perf_counter()
                previous = run(tr, episode) if stage == 1 else run(tr, episode, previous)
                sync(device)
                seconds = perf_counter()-start
                executed_training_seconds += seconds
                stage_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                _stage_checkpoint(previous, path, tr, signature, seconds, stage_peak)
                if source_cache:
                    _stage_checkpoint(previous, source_cache, tr, source_signature, seconds, stage_peak)
            training_seconds += seconds
            if stage_peak is not None:
                peak = max(peak or 0, stage_peak)
            if "history" in previous or "trace" in previous:
                write_json(previous.get("history", previous.get("trace")), directory/f"stage{stage}_history.json")
            if stage == 2:
                ranking = pd.DataFrame(previous["reliability"])
                for value, ascending in (("rho", False), ("weight", False), ("error", True), ("delta", True)):
                    if value in ranking:
                        ranking[f"{value}_rank"] = ranking[value].rank(method="min", ascending=ascending).astype(int)
                ranking.sort_values("rho", ascending=False, kind="stable").to_csv(directory/"reliability.csv", index=False)
                trace_rows = []
                for entry in previous.get("trace", []):
                    row = {key: value for key, value in entry.items() if key != "weights"}
                    row.update({f"weight_{name}": weight for name, weight in zip(
                        episode.sources, entry.get("weights", []))})
                    trace_rows.append(row)
                if trace_rows:
                    pd.DataFrame(trace_rows).to_csv(directory/"fusion_trace.csv", index=False)
            if stage == 3 and "source_alignment_mass" in previous:
                write_json(previous["source_alignment_mass"], directory/"alignment_mass.json")
            states[stage] = previous
        if not score and not validation_only and until < 4:
            return {}
        if until < 2:
            raise ValueError("Scoring RMAF requires its fused predictor or a later stage.")
        model = from_stage(tr, previous).eval().merged()
        if "merged" in previous:
            model.load_state_dict(previous["merged"])
        training_params = max(state.get("training_params", 0) for state in states.values())
        if until == 4 and not validation_only:
            save_predictor(model, episode, config, method, directory/"final_model.pt")
            write_json({k: previous[k] for k in ("anchor_strength", "parameter_drift", "max_merge_error")},
                       directory/"adaptation.json")
    else:
        path = directory/"baseline.pt"
        if path.exists():
            package = load_weights(path, device)
            state = package["model"]
            if method == "persistence":
                model = None
            elif method in {"patchtst", "timesnet", "cyclenet"}:
                from ..evaluation.modern_baselines import modern_baseline
                model = modern_baseline(config, method).to(device)
            else:
                from ..models.forecaster import PlainLSTM
                model = PlainLSTM(config["model"]).to(device)
            if model is not None:
                model.load_state_dict(state)
            training_params, history, training_seconds, peak = (package[k] for k in ("params", "history", "seconds", "peak"))
        else:
            model, training_params, history, training_seconds = run_baseline(tr, episode, method)
            executed_training_seconds += training_seconds
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            atomic_save({"model": model.state_dict() if model is not None else None, "params": training_params,
                         "history": history, "seconds": training_seconds, "peak": peak}, path)
        write_json(history, directory/"baseline_history.json")
        if model is not None and not validation_only:
            save_predictor(model, episode, config, method, directory/"final_model.pt")
    valid = episode.target_sets["validation"]
    val_rmse = tr.validation(model, [supervised_dataset(valid, model)])**0.5 if model is not None else None
    if validation_only:
        return {"validation_rmse": val_rmse}
    if not score:
        return {}
    test = episode.target_sets["test"]
    metrics, threshold = score_predictions(test, predict_batches(model, tr, test), episode, config, directory)
    times = {}
    if method == "rmaf":
        times = {f"stage{stage}_seconds": load_weights(directory/f"stage{stage}.pt")["seconds"] for stage in states}
        if until == 4:
            diagnostics(tr, episode, states[2], states[3], states[4], directory)
    costs = complexity(model, training_params, training_seconds, peak, test.x, tr)
    result = {"seed": seed, "method": method, "target": episode.target, "validation_rmse": val_rmse,
              "ramp_threshold": threshold, "metrics": metrics, "source_stage_reused": reused_source_stage, **costs, **times}
    result.update(executed_training_seconds=executed_training_seconds,
                  invocation_seconds=perf_counter()-invocation_started)
    write_json(result, directory/"metrics.json")
    return result
