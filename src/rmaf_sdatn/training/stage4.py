import torch
from torch.nn import functional as F

from ..algorithms.anchoring import anchor_loss, differences, parameter_groups, snapshot, strengths
from .common import fit_prediction
from .stage3 import from_stage


def run_stage4(tr, episode, previous):
    model = from_stage(tr, previous)
    alg = tr.config["algorithm"]
    settings = alg["anchoring"]
    reference = snapshot(model)
    lrs = tr.cfg["lr"]["target"]
    if not lrs["backbone"] < min(lrs["adapter"], lrs["evolution"]) <= max(lrs["adapter"], lrs["evolution"]) < lrs["head"]:
        raise ValueError("Target LR order must be backbone < adapter/evolution < head.")
    if settings["freeze_backbone"]:
        model.backbone.requires_grad_(False)
    groups = [{"params": module.parameters(), "lr": lrs[name]} for name, module in parameter_groups(model).items()
              if any(p.requires_grad for p in module.parameters())]
    optimizer = tr.adam(groups)
    ds = episode.target_sets["adaptation"].labeled()
    val = episode.target_sets["validation"].labeled()
    coeffs = strengths(settings, len(ds), episode.n_ref, alg["epsilon"])
    history = fit_prediction(tr, model, [ds], lambda: tr.validation(model, [val]), tr.cfg["epochs"]["target"],
                             optimizer=optimizer, penalty=lambda: anchor_loss(model, reference, coeffs), phase="stage4")
    with torch.no_grad():
        drift = {name: float(d.sqrt()) for name, d in differences(model, reference)}
        model.eval()
        merged = model.merged()
        # The deployment check is conducted before final testing and uses validation inputs only.
        x = next(iter(tr.loader(val)))[0].to(tr.device)
        before, after = model(x).power, merged(x).power
        eval_cfg = tr.config["evaluation"]
        torch.testing.assert_close(before, after, atol=eval_cfg["merge_atol"], rtol=eval_cfg["merge_rtol"])
    return {"model": model.state_dict(), "adapter_rank": previous["adapter_rank"], "history": history,
            "merged": merged.state_dict(), "max_merge_error": float((before - after).abs().max()),
            "anchor_reference": reference, "anchor_strength": coeffs, "parameter_drift": drift,
            "reliability": previous["reliability"]}
