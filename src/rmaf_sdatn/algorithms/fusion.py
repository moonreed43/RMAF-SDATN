import math
from pathlib import Path

import numpy as np
import torch

from ..serialization import atomic_save, load_weights


def project_simplex(vector):
    vector = np.asarray(vector, dtype=np.float64)
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1
    indexes = np.arange(1, len(vector) + 1)
    active = np.flatnonzero(ordered > cumulative / indexes)[-1]
    return np.maximum(vector - cumulative[active] / indexes[active], 0)


def kl_prior(weights, prior):
    mask = weights > 0
    return (weights[mask] * (weights[mask].log() - prior[mask].log())).sum()


def search(prior, objective, settings, seed, checkpoint=None):
    """Projected mutation of uniformly sampled elites, retaining incumbent scores."""
    size, generations = settings["population"], settings["generations"]
    ratio = settings["elite_ratio"]
    if size < 2 or generations < 1 or not 0 < ratio < 1:
        raise ValueError("Invalid population, number of generations or elite selection ratio.")
    if settings["mutation"] not in {"gaussian", "uniform"}:
        raise ValueError("Unknown fusion mutation.")
    rng = np.random.default_rng(seed)
    count = min(size - 1, max(1, math.ceil(ratio * size)))
    mode = settings["mode"]
    center = np.full_like(prior, 1 / len(prior)) if mode == "objective_only" else prior

    def score(vector):
        value = float(objective(vector))
        if not np.isfinite(value):
            raise ValueError("Nonfinite fusion objective.")
        return value, vector

    if mode in {"uniform", "prior_only"} or len(prior) == 1:
        vector = np.full_like(prior, 1 / len(prior)) if mode == "uniform" else center
        return vector, [{"generation": 0, "objective": score(vector)[0], "weights": vector.tolist()}]
    if mode not in {"constrained", "no_prior_penalty", "objective_only"}:
        raise ValueError(f"Unknown fusion mode: {mode}")
    trace = []
    path = Path(checkpoint) if checkpoint else None
    if path and path.exists():
        saved = load_weights(path)
        if saved["settings"] != settings or saved["seed"] != seed or not np.array_equal(saved["prior"].numpy(), prior):
            raise ValueError("Fusion checkpoint has different prior, settings or seed.")
        population = [(float(value), vector.numpy()) for value, vector in zip(saved["scores"], saved["vectors"])]
        trace = saved["trace"]
        rng.bit_generator.state = saved["rng"]
    else:
        population = [score(project_simplex(center))]
        population.extend(score(project_simplex(center + rng.normal(0, settings["init_std"], len(prior))))
                          for _ in range(size - 1))
    for generation in range(len(trace), generations):
        if generation:
            elites = sorted(population, key=lambda row: row[0])[:count]
            population = elites.copy()
            for _ in range(size - count):
                _, parent = elites[rng.integers(count)]
                std = settings["mutation_std"]
                if settings["mutation"] == "gaussian":
                    noise = rng.normal(0, std, len(prior))
                else:
                    noise = rng.uniform(-std * math.sqrt(3), std * math.sqrt(3), len(prior))
                population.append(score(project_simplex(parent + noise)))
        population.sort(key=lambda row: row[0])
        trace.append({"generation": generation + 1, "objective": population[0][0],
                      "weights": population[0][1].tolist(),
                      "population_mean": float(np.mean([row[0] for row in population]))})
        if path:
            atomic_save({"settings": settings, "seed": seed, "prior": torch.from_numpy(prior.copy()),
                         "vectors": torch.from_numpy(np.stack([row[1] for row in population])),
                         "scores": [row[0] for row in population], "trace": trace, "rng": rng.bit_generator.state}, path)
    return population[0][1], trace
