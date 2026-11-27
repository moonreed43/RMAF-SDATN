import torch

from .common import fit_prediction, param_count
from ..models.forecaster import Forecaster
from ..models.lora import LoRA


def run_stage1(tr, episode):
    cfg = tr.config["model"]
    alg = tr.config["algorithm"]
    epochs = tr.cfg["epochs"]
    model = Forecaster(cfg).to(tr.device)
    datasets = {k: ds.labeled() for k, ds in episode.sources.items()}
    validation = {k: ds.labeled() for k, ds in episode.source_validation.items()}
    history, independent = {}, {}
    if alg["independent_full_models"]:
        for name, ds in datasets.items():
            separate = Forecaster(cfg).to(tr.device)
            history[name] = fit_prediction(tr, separate, [ds], lambda: tr.validation(separate, [validation[name]]),
                                           epochs["shared"], step_limit=tr.cfg["source_batches_per_epoch"], phase=f"independent-{name}")
            independent[name] = {k: v.cpu() for k, v in separate.state_dict().items()}
        model.load_state_dict({k: torch.stack([v[k] for v in independent.values()]).mean(0) for k in model.state_dict()})
        return {"model": model.state_dict(), "adapters": {}, "history": history, "training_params": param_count(model)*len(independent)}
    history["shared"] = fit_prediction(tr, model, list(datasets.values()), lambda: tr.validation(model, validation.values()),
                                       epochs["shared"], step_limit=tr.cfg["source_batches_per_epoch"], phase="stage1-shared")
    model.requires_grad_(False)
    states = {}
    for name, ds in datasets.items():
        adapter = LoRA(model.backbone, cfg["rank"], cfg["alpha"]).to(tr.device)

        def valid():
            with torch.no_grad():
                return tr.validation(model, [validation[name]], residuals=adapter.residuals())

        def penalty():
            return alg["adapter_penalty"] * sum(residual.square().sum() for residual in adapter.residuals().values())

        history[name] = fit_prediction(tr, model, [ds], valid, epochs["adapter"], optimizer=tr.adam(adapter.parameters()),
                                       predict=lambda x: model(x, adapter.residuals()).power, penalty=penalty, tracked=[adapter],
                                       step_limit=tr.cfg["source_batches_per_epoch"], phase=f"stage1-adapter-{name}")
        states[name] = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
    return {"model": model.state_dict(), "adapters": states, "history": history,
            "training_params": param_count(model) + len(states) * param_count(adapter)}
