import numpy as np
import torch
from torch.nn import functional as F

from ..algorithms.fusion import kl_prior, search
from ..algorithms.reliability import FeatureMoments, discrepancy_statistics, reliability
from ..models.forecaster import Forecaster
from ..models.lora import LoRA, weighted_residuals


@torch.no_grad()
def run_stage2(tr, episode, previous):
    cfg, alg = tr.config["model"], tr.config["algorithm"]
    model = Forecaster(cfg).to(tr.device).eval().requires_grad_(False)
    model.load_state_dict(previous["model"])
    names = list(episode.sources)
    if not previous["adapters"]:
        return {"model": model.state_dict(), "adapter_rank": 0, "reliability": [{"source": s, "rho": 1/len(names),
                "weight": 1/len(names)} for s in names], "trace": []}
    rel_cfg, fusion_cfg = alg["reliability"], alg["fusion"]
    rel = list(tr.loader(episode.target_sets["reliability"].labeled()))
    fus = [tr.move(b) for b in tr.loader(episode.target_sets["fusion"].labeled())]

    def representations(batches):
        moments = FeatureMoments()
        for x, _ in batches:
            moments.update(model.hidden(x.to(tr.device)).mean(dim=1))
        return moments.statistics()

    target = representations(rel)
    discrepancies, errors, adapters = [], [], []
    for name in names:
        adapter = LoRA(model.backbone, cfg["rank"], cfg["alpha"]).to(tr.device)
        adapter.load_state_dict(previous["adapters"][name])
        adapters.append(adapter)
        residuals = adapter.residuals()
        features = representations(tr.loader(episode.sources[name].labeled()))
        discrepancies.append(discrepancy_statistics(features, target, rel_cfg["covariance"]))
        errors.append(sum(F.l1_loss(model(x, residuals).power, y, reduction="sum") for x, y in map(tr.move, rel))
                      / sum(y.numel() for _, y in rel))
    delta, error = torch.stack(discrepancies), torch.stack(errors)
    prior, norm_delta, norm_error = reliability(delta, error, rel_cfg["eta"], rel_cfg["temperature"], alg["epsilon"])
    stacked = {key: torch.stack([a.matrices[key]() for a in adapters]) for key in adapters[0].matrices}
    denominator = sum(y.numel() for _, y in fus)

    def objective(vector):
        weights = torch.as_tensor(vector, device=tr.device, dtype=prior.dtype)
        residuals = weighted_residuals(stacked, weights.float())
        value = sum(F.mse_loss(model(x, residuals).power, y, reduction="sum") for x, y in fus) / denominator
        if fusion_cfg["mode"] in {"constrained", "prior_only", "uniform"}:
            value = value + fusion_cfg["prior_penalty"] * kl_prior(weights, prior)
        return value

    checkpoint = tr.progress_dir/"stage2-search.pt" if tr.progress_dir else None
    weights, trace = search(prior.cpu().numpy(), objective, fusion_cfg, tr.seed, checkpoint)
    adapter = LoRA.fuse(model.backbone, adapters, weights)
    model.adapter = adapter
    rows = [{"source": name, "delta": float(delta[i]), "error": float(error[i]),
             "normalized_delta": float(norm_delta[i]), "normalized_error": float(norm_error[i]),
             "rho": float(prior[i]), "weight": float(weights[i])} for i, name in enumerate(names)]
    return {"model": model.state_dict(), "adapter_rank": adapter.rank, "reliability": rows, "trace": trace}
