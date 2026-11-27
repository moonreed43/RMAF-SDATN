import math

import torch


def parameter_groups(model):
    groups = {"backbone": model.backbone, "head": model.head}
    if getattr(model, "evolution", None) is not None:
        groups["evolution"] = model.evolution
    if getattr(model, "adapter", None) is not None:
        groups["adapter"] = model.adapter
    return groups


def snapshot(model):
    result = {}
    for group, module in parameter_groups(model).items():
        params = module.residuals() if group == "adapter" else dict(module.named_parameters())
        result[group] = {name: value.detach().clone() for name, value in params.items()}
    return result


def differences(model, reference):
    for group, module in parameter_groups(model).items():
        params = module.residuals() if group == "adapter" else dict(module.named_parameters())
        terms = [(value - reference[group][name]).square().sum() for name, value in params.items()]
        yield group, torch.stack(terms).sum()


def strengths(settings, n, n_ref, epsilon):
    if n <= 0 or n_ref <= 0:
        raise ValueError("Anchoring requires positive valid sample counts.")
    mode = settings["mode"]
    if mode not in {"none", "fixed", "scarcity"}:
        raise ValueError(f"Unknown anchoring mode: {mode}")
    scale = {"none": 0.0, "fixed": 1.0, "scarcity": 1 / math.sqrt(n/n_ref + epsilon)}[mode]
    return {k: settings[k] * settings["multiplier"] * scale for k in ("backbone", "adapter", "evolution")}


def anchor_loss(model, reference, weights):
    total = next(model.parameters()).new_zeros(())
    for group, value in differences(model, reference):
        if weights.get(group, 0):
            total = total + weights[group] * value
    return total
