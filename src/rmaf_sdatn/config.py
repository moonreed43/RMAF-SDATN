from copy import deepcopy
from pathlib import Path

import yaml


def merge(base, overrides):
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path, _parents=()):
    path = Path(path).resolve()
    if path in _parents:
        chain = " -> ".join(str(p) for p in (*_parents, path))
        raise ValueError(f"Circular YAML include: {chain}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    result = {}
    for include in config.pop("include", []):
        result = merge(result, load_config(path.parent / include, (*_parents, path)))
    return merge(result, config)
