import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .data.download import china, pvdaq
from .data.preprocess import preprocess
from .data.splits import StationData, build_episode, source_ids, test_windows
from .evaluation.metrics import predict_batches, score_predictions
from .experiments import experiments
from .inference import load_predictor, predict_csv
from .serialization import write_json
from .training.common import Training, configure
from .training.runner import run_trial
from .validation import validate_config


def parser(command):
    p = argparse.ArgumentParser(prog=f"{command}.py")
    if command not in {"predict", "evaluate", "summarize"}:
        p.add_argument("--config", default=f"configs/experiments/{'pvdaq' if command == 'download_pvdaq' else 'main'}.yaml")
    if command == "download_data":
        p.add_argument("--dataset", choices=["all", "pvod", "state_grid"], default="all")
    if command == "check_project":
        p.add_argument("--output", type=Path, default=Path("outputs/project_audit.json"))
    if command == "summarize":
        p.add_argument("--root", required=True, type=Path)
        p.add_argument("--config", help="Optional experiment YAML to regenerate figures")
    if command in {"download_pvdaq", "preprocess"}:
        p.add_argument("--stations", nargs="+")
    if command == "download_pvdaq":
        p.add_argument("--download", action="store_true")
        p.add_argument("--match", help="Override the case-insensitive archive object regex")
        p.add_argument("--max-bytes", type=int)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--chunk-mb", type=int, default=8)
    if command in {"train", "build_splits"}:
        p.add_argument("--target", required=True)
        p.add_argument("--transfer", choices=["mixed", "intra", "inter", "external"], default="mixed")
        p.add_argument("--sources", nargs="+", help="Explicit subset for diagnostic runs")
        p.add_argument("--budget", type=int, default=30)
        p.add_argument("--window", default="final")
    if command == "train":
        p.add_argument("--seed", type=int)
        p.add_argument("--run-dir", required=True, type=Path)
        p.add_argument("--until-stage", type=int, choices=[1, 2, 3, 4], default=4)
    if command == "build_splits":
        p.add_argument("--output", type=Path)
    if command == "run_experiments":
        p.add_argument("--targets", nargs="+")
        p.add_argument("--methods", nargs="+")
        p.add_argument("--variants", nargs="+")
        p.add_argument("--sources", nargs="+", help="Explicit subset for diagnostic runs")
        p.add_argument("--budgets", nargs="+", type=int, choices=[1, 3, 7, 14, 30])
        p.add_argument("--seeds", nargs="+", type=int)
        p.add_argument("--transfers", nargs="+", choices=["mixed", "intra", "inter", "external"])
        p.add_argument("--keep-going", action="store_true", help="Record failed data/training cases and continue independent trials")
    if command == "evaluate":
        p.add_argument("--run-dir", required=True, type=Path)
        p.add_argument("--checkpoint", type=Path)
        p.add_argument("--device", default="cpu")
    if command == "predict":
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--input", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--timezone")
        p.add_argument("--station")
        p.add_argument("--capacity", type=float)
        p.add_argument("--native-minutes", type=int, help="Optional finer input cadence to aggregate into complete model intervals")
    return p


def evaluate(args):
    directory = args.run_dir
    cfg = load_config(directory/"config.yaml")
    man = json.loads((directory/"split.json").read_text(encoding="utf-8"))
    identity = json.loads((directory/"identity.json").read_text(encoding="utf-8"))
    target = man["target"]
    test_time = man["target_sets"]["test"]
    dev_end = man["target_sets"]["validation"]["end_exclusive"]
    times = (pd.Timestamp(test_time["start"]), pd.Timestamp(test_time["end_exclusive"]), pd.Timestamp(dev_end))
    episode = build_episode(StationData(cfg), target, list(man["source_sets"]), man["budget_days"], times)
    if man != episode.manifest:
        raise ValueError("Processed data or split has changed since training.")
    cfg["training"]["device"] = args.device
    tr = Training(cfg, identity["seed"], configure(identity["seed"], cfg["training"]))
    if identity["method"] == "persistence" and args.checkpoint is None:
        model = None
    else:
        model, payload = load_predictor(args.checkpoint or directory/"final_model.pt", tr.device)
        if payload["station"]["station"] != target:
            raise ValueError("Checkpoint and split refer to different target stations.")
    windows = episode.target_sets["test"]
    output = directory/"reevaluation"
    output.mkdir(exist_ok=True)
    scores, threshold = score_predictions(windows, predict_batches(model, tr, windows), episode, cfg, output)
    write_json({"metrics": scores, "ramp_threshold": threshold}, output/"metrics.json")
    print(output)


def main(command):
    args = parser(command).parse_args()
    if command == "predict":
        result = predict_csv(args.checkpoint, args.input, args.output, device=args.device, timezone=args.timezone,
                             station=args.station, capacity=args.capacity, native_minutes=args.native_minutes)
        print(result.to_string(index=False))
        return
    if command == "evaluate":
        evaluate(args)
        return
    if command == "summarize":
        from .experiments import recover_results
        frame = recover_results(args.root)
        if frame is None:
            raise ValueError(f"No completed trials found in {args.root}.")
        if args.config:
            from .evaluation.visualization import plot_experiment
            plot_experiment(frame, load_config(args.config), args.root/"figures")
        print(f"{len(frame)} metric rows recovered at {args.root}")
        return
    cfg = load_config(args.config)
    validate_config(cfg)
    d, exp = cfg["data"], cfg["experiment"]
    if command == "check_project":
        from .audit import audit_project
        report = audit_project(cfg, args.output)
        print(f"{report['ready_episodes']} ready episodes, {report['unavailable_episodes']} unavailable; {args.output}")
    elif command == "download_data":
        china(d, args.dataset)
    elif command == "download_pvdaq":
        pvdaq(d, args.stations or exp["targets"], args.download, args.match, args.max_bytes, args.workers, args.chunk_mb)
    elif command == "preprocess":
        stations = args.stations or [k for k, v in d["stations"].items()
                                      if v["dataset"] != "pvdaq" or k in exp["targets"]]
        for station in stations:
            result = preprocess(station, d, cfg["model"])
            print(f"{station}: {result['retained']} valid observations, {result['night_zero']} zeros")
    elif command == "run_experiments":
        if args.keep_going:
            cfg["experiment"]["keep_going"] = True
        print(experiments(cfg, args.targets, args.methods, args.variants, args.sources,
                          args.budgets, args.transfers, args.seeds))
    else:
        store = StationData(cfg)
        sources = args.sources or source_ids(cfg, args.target, args.transfer)
        missing = {}
        seasons = test_windows(store, args.target, failures=missing)
        if args.window not in seasons:
            reason = missing.get(args.window, f"Unknown window {args.window}. Available windows: {list(seasons)}")
            raise ValueError(reason)
        times = seasons[args.window]
        episode = build_episode(store, args.target, sources, args.budget, times)
        if command == "build_splits":
            output = args.output or Path(d["root"])/"splits"/f"{args.target}-{args.window}-{args.transfer}-{args.budget}.json"
            write_json(episode.manifest, output)
            print(output)
        else:
            seed = cfg["training"]["seeds"][0] if args.seed is None else args.seed
            result = run_trial(cfg, episode, seed, args.run_dir, until=args.until_stage, score=args.until_stage == 4)
            if result:
                record = {key: value for key, value in result.items() if key != "metrics"}
                record.update(window=args.window, transfer=args.transfer, budget=args.budget, variant="full",
                              source_case="all", sources=" ".join(sources), incompatible_count=None,
                              path=str(args.run_dir))
                write_json({"record": record, "metrics": result["metrics"]}, args.run_dir/"experiment_record.json")
                overall = next(row for row in result["metrics"] if row["subset"] == "all")
                print(f"{args.target}: RMSE={overall['rmse']:.6f}, MAE={overall['mae']:.6f}, "
                      f"R2={overall['r2']}; {args.run_dir}")
