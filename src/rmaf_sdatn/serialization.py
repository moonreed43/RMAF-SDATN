import json
import os
from pathlib import Path
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

import numpy as np
import torch


@contextmanager
def atomic_destination(filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=path.name+".", suffix=".tmp", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_save(payload, filename):
    with atomic_destination(filename) as temporary:
        torch.save(payload, temporary)


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def json_value(value):
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    with atomic_destination(path) as temporary:
        temporary.write_text(json.dumps(json_value(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_weights(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def write_csv(frame, filename, index=False):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_destination(path) as temporary:
        frame.to_csv(temporary, index=index)
