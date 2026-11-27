from time import perf_counter

import torch
from torch.nn import functional as F
from torch.utils.data import ConcatDataset

from ..algorithms.alignment import adversarial_strength, coral, domain_loss, mmd
from ..models.forecaster import DomainDiscriminator, PlainLSTM
from ..models.cyclenet import supervised_dataset
from ..training.common import RepeatBatches, fit_prediction, param_count
from .complexity import sync

METHODS = {"persistence", "target_only", "multi_source", "dann", "mmd", "coral",
           "timesnet", "patchtst", "cyclenet"}


def run_baseline(tr, episode, method):
    if method not in METHODS:
        raise ValueError(f"Unknown baseline: {method}")
    if method == "persistence":
        return None, 0, {}, 0.0
    cfg, alg = tr.config["model"], tr.config["algorithm"]
    epochs = tr.cfg["epochs"]
    if method in {"timesnet", "patchtst", "cyclenet"}:
        from .modern_baselines import modern_baseline
        model = modern_baseline(tr.config, method).to(tr.device)
    else:
        model = PlainLSTM(cfg).to(tr.device)
    history = {}
    sync(tr.device)
    start = perf_counter()
    src = {name: supervised_dataset(win, model) for name, win in episode.sources.items()}
    s_val = [supervised_dataset(win, model) for win in episode.source_validation.values()]
    t_val = [supervised_dataset(episode.target_sets["validation"], model)]
    cap = tr.cfg["source_batches_per_epoch"]
    if method != "target_only":
        history["shared"] = fit_prediction(tr, model, list(src.values()), lambda: tr.validation(model, s_val),
                                           epochs["shared"], step_limit=cap, phase="baseline-shared")
    training_params = param_count(model)
    if method in {"dann", "mmd", "coral"}:
        discriminator = None
        tracked = [model]
        optimizer_groups = [{"params": model.parameters()}]
        if method == "dann":
            discriminator = DomainDiscriminator(cfg["hidden"], cfg["discriminator_hidden"]).to(tr.device)
            optimizer_groups.append({"params": discriminator.parameters()})
            tracked.append(discriminator)
            training_params += param_count(discriminator)
        optimizer = tr.adam(optimizer_groups)
        src_loaders, num_steps = tr.balanced(src)
        tgt_loader = tr.loader(episode.target_sets["adaptation"].unlabeled(), shuffle=True)

        def adapt(epoch):
            model.train()
            if discriminator is not None:
                discriminator.train()
            src_batches = [RepeatBatches(loader) for loader in src_loaders.values()]
            tgt_batches = RepeatBatches(tgt_loader)
            reports = []
            for step in range(num_steps):
                pairs = [tr.move(batch.next()) for batch in src_batches]
                x_target = tgt_batches.next().to(tr.device)
                target = model(x_target).shared
                outputs = [model(x) for x, _ in pairs]
                pred = torch.stack([F.mse_loss(out.power, y) for out, (_, y) in zip(outputs, pairs)]).mean()
                source = torch.cat([out.shared for out in outputs])
                acc = pred.detach().new_tensor(float("nan"))
                if method == "dann":
                    loss, acc = domain_loss(discriminator, source, target, source.new_ones(len(source)))
                elif method == "mmd":
                    loss = mmd(source, target, alg["epsilon"])
                else:
                    loss = coral(source, target)
                progress = (epoch*num_steps+step) / max(1, epochs["adversarial"]*num_steps-1)
                settings = alg["alignment"]
                strength = adversarial_strength(progress, settings["lambda_max"], settings["gamma"])
                tr.step(pred + alg["baseline_alignment_weight"] * strength * loss, optimizer)
                reports.append(torch.stack((pred.detach(), loss.detach(), acc)))
            values = torch.stack(reports).mean(0).tolist()
            return dict(zip(("source_prediction_mse", "domain_loss", "domain_balanced_accuracy"), values))

        history["alignment"] = tr.early_fit(epochs["adversarial"], tracked, adapt, lambda: tr.validation(model, t_val),
                                            optimizer, "baseline-alignment")
    target_training = ConcatDataset([supervised_dataset(episode.target_sets[key], model)
                                    for key in ("reliability", "fusion", "adaptation")])
    history["target"] = fit_prediction(tr, model, [target_training], lambda: tr.validation(model, t_val),
                                       epochs["target"], phase="baseline-target")
    sync(tr.device)
    return model.eval(), training_params, history, perf_counter()-start
