import torch
from torch import nn
from torch.nn import functional as F

from ..algorithms.alignment import adversarial_strength, alignment_features, domain_loss, sample_gates
from ..algorithms.anchoring import parameter_groups
from ..models.forecaster import DomainDiscriminator, Forecaster
from .common import RepeatBatches, param_count


def from_stage(tr, stage):
    model = Forecaster(tr.config["model"], stage["adapter_rank"]).to(tr.device)
    model.load_state_dict(stage["model"])
    return model


def run_stage3(tr, episode, previous):
    model = from_stage(tr, previous)
    alg, cfg = tr.config["algorithm"], tr.config["model"]
    a = alg["alignment"]
    names = list(episode.sources)
    if not a["enabled"]:
        return {**previous, "history": [], "discriminators": {}, "training_params": param_count(model)}
    dim = cfg["shared_dim"] + (cfg["supplementary_dim"] if a["branch"] == "both" else 0)
    global_disc = a["discriminator"] == "global"
    discriminators = nn.ModuleDict({name: DomainDiscriminator(dim, cfg["discriminator_hidden"]) for
                                   name in (["global"] if global_disc else names)}).to(tr.device)
    lrs = tr.cfg["lr"]["adversarial"]
    groups = [{"params": module.parameters(), "lr": lrs[name]} for name, module in parameter_groups(model).items()]
    groups.append({"params": discriminators.parameters(), "lr": lrs["discriminator"]})
    optimizer = tr.adam(groups)
    src, num_steps = tr.balanced({s: ds.labeled() for s, ds in episode.sources.items()})
    target = tr.loader(episode.target_sets["adaptation"].unlabeled(), shuffle=True)
    validation = [episode.target_sets["validation"].labeled()]
    epochs = tr.cfg["epochs"]["adversarial"]
    rho = {row["source"]: row["rho"] for row in previous["reliability"]}
    station_weight = rho if a["station_weights"] == "reliability" else {k: 1/len(names) for k in names}

    def train_epoch(epoch):
        model.train()
        discriminators.train()
        batches = {s: RepeatBatches(src[s]) for s in names}
        target_batches = RepeatBatches(target)
        reports = []
        mass = []
        for step in range(num_steps):
            progress = (epoch * num_steps + step) / max(epochs * num_steps - 1, 1)
            weight = adversarial_strength(progress, a["lambda_max"], a["gamma"])
            x_target = target_batches.next().to(tr.device, non_blocking=True)
            source_pairs = [tr.move(batches[name].next()) for name in names]
            lengths = [len(x_target), *(len(x) for x, _ in source_pairs)]
            joint = model(torch.cat([x_target, *(x for x, _ in source_pairs)]))
            parts = zip(*(value.split(lengths) for value in joint))
            target_result = type(joint)(*next(parts))
            target_features = alignment_features(target_result, a["branch"])
            prediction, domain, accuracy = [], [], []
            source_mass = []
            for name, (x, y), result in zip(names, source_pairs, parts):
                out = type(joint)(*result)
                if a["gate"] == "none":
                    gates = out.shared.new_ones(len(x))
                else:
                    gates, _ = sample_gates(out.shared, target_result.shared, a["gate_range"], alg["epsilon"])
                discriminator = discriminators["global" if global_disc else name]
                dloss, acc = domain_loss(discriminator, alignment_features(out, a["branch"]), target_features,
                                         gates, a["gate"], a["gate_fraction"])
                prediction.append(rho[name] * F.mse_loss(out.power, y))
                domain.append(station_weight[name] * dloss)
                accuracy.append(acc)
                # Eq. (18) assigns this total coefficient to the source BCE.
                mean_gate = gates.new_tensor(1.) if a["gate"] in {"high", "low"} else gates.mean()
                source_mass.append(weight * station_weight[name] * mean_gate)
            pred_loss = torch.stack(prediction).sum()
            dom_loss = torch.stack(domain).sum()
            tr.step(pred_loss + weight * dom_loss, optimizer)
            reports.append(torch.stack((pred_loss.detach(), dom_loss.detach(), torch.stack(accuracy).mean())))
            mass.append(torch.stack(source_mass))
        pred, domain, acc = torch.stack(reports).mean(dim=0).tolist()
        station_mass = torch.stack(mass).mean(dim=0).tolist()
        return {"source_prediction_mse": pred, "domain_loss": domain, "domain_balanced_accuracy": acc,
                "source_alignment_mass": dict(zip(names, station_mass))}

    history = tr.early_fit(epochs, [model, discriminators], train_epoch, lambda: tr.validation(model, validation),
                           optimizer, "stage3")
    best = min(history, key=lambda row: row["validation_mse"])
    return {**previous, "model": model.state_dict(), "history": history, "discriminators": discriminators.state_dict(),
            "source_alignment_mass": best["source_alignment_mass"],
            "training_params": param_count(model) + param_count(discriminators)}
