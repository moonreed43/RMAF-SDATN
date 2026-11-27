import numpy as np
import pandas as pd
import torch

from ..algorithms.alignment import alignment_features, mmd, sample_gates
from ..models.forecaster import DomainDiscriminator
from ..training.stage3 import from_stage


def balanced_inputs(episode, count):
    domains = {name: windows.x for name, windows in episode.source_validation.items()}
    domains["target"] = episode.target_sets["test"].x
    size = min(count, *(len(x) for x in domains.values()))
    return {name: x[np.linspace(0, len(x)-1, size, dtype=int)] for name, x in domains.items()}


@torch.inference_mode()
def diagnostics(tr, episode, before, adversarial, after, directory):
    cfg = tr.config["evaluation"]
    alg = tr.config["algorithm"]
    samples = balanced_inputs(episode, cfg["diagnostic_samples"])
    stage_features, gates, distances, attention = [], [], [], []
    classifiers = adversarial["discriminators"]
    align = alg["alignment"]
    model_cfg = tr.config["model"]
    dimension = model_cfg["shared_dim"] + (model_cfg["supplementary_dim"] if align["branch"] == "both" else 0)
    discs = {}
    if classifiers:
        for key in (["global"] if align["discriminator"] == "global" else list(episode.sources)):
            disc = DomainDiscriminator(dimension, model_cfg["discriminator_hidden"]).to(tr.device).eval()
            disc.load_state_dict({k[len(key)+1:]: v for k, v in classifiers.items() if k.startswith(key+".")})
            discs[key] = disc
    rho = {row["source"]: row["rho"] for row in before["reliability"]}
    for label, stage in (("fused", before), ("adversarial", adversarial), ("adapted", after)):
        model = from_stage(tr, stage).eval()
        mapped = {}
        for key, x in samples.items():
            values = [model(chunk.to(tr.device)) for chunk in torch.from_numpy(x).split(tr.cfg["batch_size"])]
            mapped[key] = type(values[0])(*(torch.cat(items, dim=0) for items in zip(*values)))
        target = mapped["target"]
        for source, out in mapped.items():
            if source == "target":
                continue
            record = {"stage": label, "source": source, "shared_mmd": float(mmd(out.shared, target.shared, alg["epsilon"])),
                      "supplementary_mmd": float(mmd(out.supplementary, target.supplementary, alg["epsilon"]))}
            if label == "adversarial":
                s_features = alignment_features(out, align["branch"])
                t_features = alignment_features(target, align["branch"])
                key = "global" if align["discriminator"] == "global" else source
                if key in discs:
                    disc = discs[key]
                    record["domain_balanced_accuracy"] = float(((disc(s_features) >= 0).float().mean()
                                                               + (disc(t_features) < 0).float().mean())/2)
                for batch_index, (s, t) in enumerate(zip(out.shared.split(tr.cfg["batch_size"]), target.shared.split(tr.cfg["batch_size"]))):
                    g, similarity = sample_gates(s, t, align["gate_range"], alg["epsilon"])
                    applied = g.clone()
                    if not align["enabled"]:
                        applied.zero_()
                    elif align["gate"] == "none":
                        applied.fill_(1)
                    elif align["gate"] in {"high", "low"}:
                        count = max(1, round(len(g)*align["gate_fraction"]))
                        selected = g.argsort(descending=align["gate"] == "high", stable=True)[:count]
                        applied.zero_()
                        applied[selected] = len(g)/count
                    weight = rho[source] if align["station_weights"] == "reliability" else 1/len(rho)
                    gates.extend({"source": source, "sample": batch_index*tr.cfg["batch_size"]+index,
                                  "gate": float(v), "applied_gate": float(actual), "similarity": float(u),
                                  "rho_times_gate": float(actual)*weight}
                                 for index, (v, actual, u) in enumerate(zip(g.cpu(), applied.cpu(), similarity.cpu())))
            distances.append(record)
        for name, out in mapped.items():
            for branch, tensor in (("shared", out.shared), ("supplementary", out.supplementary)):
                stage_features.append((label, name, branch, tensor.cpu().numpy()))
        if label == "adapted":
            attn = target.attention.cpu().numpy()
            magnitudes = target.magnitudes.cpu().numpy()
            for sample in range(len(attn)):
                for lag in range(attn.shape[1]):
                    row = {"sample": sample, "position": lag+1, "attention": attn[sample, lag]}
                    row.update({f"magnitude_{s}": magnitudes[sample, lag, i] for i, s in enumerate(model_cfg["scales"])})
                    attention.append(row)

    records = []
    for branch in ("shared", "supplementary"):
        group = [(label, name, x) for label, name, b, x in stage_features if b == branch]
        arr = np.concatenate([x for _, _, x in group], axis=0)
        center = arr.mean(axis=0)
        _, _, axes = np.linalg.svd(arr-center, full_matrices=False)
        for label, name, x in group:
            xy = (x-center) @ axes[:2].T
            records.extend({"stage": label, "domain": name, "branch": branch, "pc1": row[0],
                            "pc2": row[1] if xy.shape[1] > 1 else 0} for row in xy)
    pd.DataFrame(records).to_csv(directory/"features.csv", index=False)
    pd.DataFrame(distances).to_csv(directory/"distances.csv", index=False)
    pd.DataFrame(gates).to_csv(directory/"gates.csv", index=False)
    pd.DataFrame(attention).to_csv(directory/"attention.csv", index=False)
