"""Render plots solely from saved run records and model diagnostics."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator
import numpy as np
import pandas as pd

BLUE = "#1f6f5f"
ORANGE = "#b65424"
GRAY = "#606c79"
METHOD_ORDER = ("rmaf", "persistence", "target_only", "multi_source", "dann", "mmd", "coral",
                "patchtst", "timesnet", "cyclenet")
METHOD_LABELS = {"rmaf": "RMAF-SDATN", "target_only": "Target-only LSTM",
                 "persistence": "Persistence", "multi_source": "Multi-source + FT",
                 "dann": "DANN", "mmd": "MMD-Transfer", "coral": "DeepCORAL",
                 "timesnet": "TimesNet", "patchtst": "PatchTST", "cyclenet": "CycleNet"}


def figure(rows=1, columns=1, size=(6.7, 3.2)):
    return plt.subplots(rows, columns, figsize=size, layout="constrained", squeeze=False)


def save(fig, file):
    file.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.grid(True, axis="y", color="#dfe4e8", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.title.set_fontsize(11)
    fig.savefig(file.with_suffix(".svg"))
    fig.savefig(file.with_suffix(".png"), dpi=200)
    plt.close(fig)


def _bar(ax, frame, value, xlabel, color=BLUE, order_by=None):
    frame = frame.sort_values(order_by or value)
    ax.barh(frame.source.astype(str), frame[value], color=color, height=0.65)
    ax.set_xlabel(xlabel)


def plot_trial(directory, config, figures):
    pred = pd.read_csv(directory/"predictions.csv")
    step = max(config["evaluation"].get("steps", [pred.step.min()]))
    pred = pred[pred.step == step].copy()
    pred["timestamp"] = pd.to_datetime(pred.timestamp, utc=True)
    focus = pred.timestamp.iloc[0] + pd.Timedelta(days=config["evaluation"]["figure_days"])
    sample = pred[pred.timestamp < focus]
    fig, (axes,) = figure()
    ax = axes[0]
    ax.plot(sample.timestamp, sample.actual, label="Observed", color=GRAY, lw=1.2)
    ax.plot(sample.timestamp, sample.prediction, label="Forecast", color=BLUE, lw=1.6, ls="--")
    ax.set(ylabel="Power / installed capacity", xlabel="Target timestamp (UTC)", title=f"Test window · lead step {step}")
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(HourLocator(byhour=(0, 12)))
    ax.xaxis.set_major_formatter(DateFormatter("%b %d\n%H:%M"))
    save(fig, figures/"forecast")
    if "strong_ramp" in pred and pred.strong_ramp.any():
        fig, (axes,) = figure()
        ax = axes[0]
        subset = pred[pred.strong_ramp]
        ax.scatter(subset.actual, subset.prediction, color=ORANGE, alpha=0.55, s=13, label=f"Strong ramp (n={len(subset)})")
        low, high = min(subset.actual.min(), subset.prediction.min()), max(subset.actual.max(), subset.prediction.max())
        ax.plot([low, high], [low, high], color=GRAY, ls="--", lw=1.0, label="Perfect forecast")
        ax.set(xlabel="Observed normalized power", ylabel="Forecast normalized power")
        ax.legend(frameon=False)
        save(fig, figures/"strong_ramp")
    rel_path = directory/"reliability.csv"
    if rel_path.exists():
        rel = pd.read_csv(rel_path)
        fig, (axes,) = figure(1, 2, (7.0, 3.1))
        _bar(axes[0], rel, "rho", "Reliability prior")
        _bar(axes[1], rel, "weight", "Optimized fusion weight", GRAY, order_by="rho")
        save(fig, figures/"source_weights")
    gate_path = directory/"gates.csv"
    if gate_path.exists() and gate_path.stat().st_size > 3:
        gates = pd.read_csv(gate_path)
        fig, (axes,) = figure()
        ax = axes[0]
        g = list(gates.groupby("source").gate)
        ax.boxplot([series.values for _, series in g], medianprops={"color": BLUE, "linewidth": 1.2})
        ax.set_xticks(np.arange(1, len(g)+1), [str(name) for name, _ in g])
        ax.set(ylabel="Sample gate", xlabel="Source station", title="Gate distribution in post-training diagnostics")
        save(fig, figures/"sample_gates")
        frame = pd.read_csv(directory/"features.csv")
        frame = frame[(frame.branch == "shared") & frame.stage.isin(["fused", "adversarial"])]
        def limits(values):
            margin = max((values.max()-values.min())*0.04, 1e-8)
            return values.min()-margin, values.max()+margin

        fig, (axes,) = figure(1, 2, (7.2, 3.1))
        for axis, stage, title in zip(axes, ("fused", "adversarial"), ("Before alignment", "After alignment")):
            subset = frame[frame.stage == stage]
            for target, name, color, marker in ((False, "Sources", GRAY, "."), (True, "Target", BLUE, "x")):
                values = subset[(subset.domain == "target") == target]
                axis.scatter(values.pc1, values.pc2, color=color, label=name, marker=marker, alpha=0.5, s=12)
            axis.set(title=title, xlabel="Common PC 1", ylabel="Common PC 2",
                     xlim=limits(frame.pc1), ylim=limits(frame.pc2))
        axes[1].legend(frameon=False)
        save(fig, figures/"feature_alignment")
    attn_path = directory/"attention.csv"
    if attn_path.exists():
        attention = pd.read_csv(attn_path).groupby("position").mean(numeric_only=True)
        fig, (axes,) = figure(1, 2, (7.1, 3.0))
        axes[0].plot(attention.index, attention.attention, color=BLUE, marker="o", ms=3)
        axes[0].set(xlabel="Historical position", ylabel="Mean attention weight")
        for style, color, name in zip(("-", "--", "-."), ("#353a38", "#6d7370", "#8c9290"),
                                      [c for c in attention if c.startswith("magnitude_")]):
            axes[1].plot(attention.index, attention[name], label=f"Lag {name.split('_')[-1]}", ls=style, color=color)
        axes[1].set(xlabel="Historical position", ylabel="Mean hidden-state change (L2)")
        axes[1].legend(frameon=False)
        save(fig, figures/"temporal_attention")
    adapt = directory/"adaptation.json"
    if adapt.exists():
        import json
        drift = json.loads(adapt.read_text(encoding="utf-8"))["parameter_drift"]
        fig, (axes,) = figure()
        ax = axes[0]
        ax.bar(list(drift), list(drift.values()), color=BLUE)
        ax.set(ylabel="L2 displacement from Stage 3", xlabel="Parameter group")
        save(fig, figures/"adaptation_drift")


def plot_experiment(frame, config, root):
    full = frame[(frame.subset == "all") & (frame.source_case == "all")]
    if full.empty:
        return
    for transfer, data in full.groupby("transfer"):
        step = data.step.max()
        data = data[data.step == step]
        selected = data[data.variant == "full"]
        if not selected.empty:
            per_seed = selected.groupby(["target", "method", "budget", "seed"]).rmse.mean().reset_index()
            for target, curves in per_seed.groupby("target"):
                if curves.budget.nunique() == 1:
                    stats = curves.groupby("method").rmse.agg(["mean", "std", "count"])
                    order = [name for name in METHOD_ORDER if name in stats.index]
                    order.extend(sorted(set(stats.index)-set(order)))
                    stats = stats.reindex(order)
                    fig, (axes,) = figure(size=(6.7, max(2.5, .34*len(order)+1.0)))
                    ax = axes[0]
                    for position, (name, values) in enumerate(stats.iterrows()):
                        ax.errorbar(values["mean"], position,
                                    xerr=values["std"] if values["count"] > 1 else None,
                                    fmt="o", color=BLUE if name == "rmaf" else GRAY, ms=5, capsize=3)
                    ax.set_yticks(range(len(order)),
                                  [f"{METHOD_LABELS.get(name, name)} (n={int(stats.loc[name, 'count'])})" for name in order])
                    ax.invert_yaxis()
                    ax.set(xlabel="RMSE (normalized power; lower is better)",
                           title=f"{target} · {transfer} · {curves.budget.iloc[0]} days · step {step}")
                    ax.grid(False, axis="y")
                    save(fig, root/transfer/f"budget_{target}")
                    continue
                names = [name for name in METHOD_ORDER if name in set(curves.method)]
                names.extend(sorted(set(curves.method)-set(names)))
                focal = ["rmaf"] if "rmaf" in names else []
                others = [name for name in names if name not in focal]
                capacity = 5-len(focal)
                panels = ([names] if len(names) <= 5 else
                          [focal+others[i:i+capacity] for i in range(0, len(others), capacity)])
                columns = min(2, len(panels))
                rows_count = (len(panels)+columns-1)//columns
                fig, axes = figure(rows_count, columns, (6.7, 3.2*rows_count))
                for ax, panel in zip(axes.flat, panels):
                    for method in panel:
                        stats = curves[curves.method == method].groupby("budget").rmse.agg(["mean", "std", "count"])
                        index = METHOD_ORDER.index(method) if method in METHOD_ORDER else names.index(method)
                        color = BLUE if method == "rmaf" else ("#353a38", "#686e6b", "#8c9290")[(index-1)%3]
                        style = "-" if method == "rmaf" else ("-", "--", "-.")[(index-1)//3%3]
                        marker = "o" if method == "rmaf" else ("s", "^", "D", "v")[(index-1)%4]
                        low, high = stats["count"].min(), stats["count"].max()
                        count = str(low) if low == high else f"{low}–{high}"
                        label = f"{METHOD_LABELS.get(method, method)} (n={count})"
                        ax.plot(stats.index, stats["mean"], color=color, marker=marker, ls=style,
                                lw=1.6 if method == "rmaf" else 1.1, ms=3.5, label=label)
                        known = stats[stats["count"] > 1]
                        if not known.empty:
                            ax.errorbar(known.index, known["mean"], yerr=known["std"], color=color,
                                        capsize=3, fmt="none", alpha=.7)
                    ax.set(ylabel="RMSE (normalized power)", xlabel="Target development budget (days)",
                           title=f"{target} · {transfer} · step {step}")
                    ax.set_xticks(sorted(curves.budget.unique()))
                    ax.legend(frameon=False, fontsize=8, ncol=1)
                for ax in list(axes.flat)[len(panels):]:
                    fig.delaxes(ax)
                # Identical limits preserve comparisons across baseline facets.
                limits = [ax.get_ylim() for ax in fig.axes]
                for ax in fig.axes:
                    ax.set_ylim(min(a for a, b in limits), max(b for a, b in limits))
                save(fig, root/transfer/f"budget_{target}")
        variants = data[data.method == "rmaf"].groupby(["variant", "budget"]).rmse.mean().reset_index()
        if variants.variant.nunique() > 1:
            variants = variants[variants.budget == variants.budget.max()].rename(columns={"variant": "source"})
            fig, (axes,) = figure(size=(6.8, max(2.5, 0.28*len(variants)+1)))
            variants["source"] = variants.source.str.replace("_", " ").replace({"full": "RMAF-SDATN"})
            variants = variants.sort_values("rmse")
            axes[0].barh(variants.source, variants.rmse,
                         color=[BLUE if name == "RMAF-SDATN" else GRAY for name in variants.source], height=.65)
            axes[0].set_xlabel("Mean test RMSE (normalized power)")
            save(fig, root/transfer/"ablations")
        for target, rows in data[(data.method == "rmaf") & (data.variant == "full")].groupby("target"):
            row = rows.sort_values(["budget", "window", "seed"], ascending=[False, True, True]).iloc[0]
            plot_trial(Path(row.path), config, root/transfer/f"target_{target}")
