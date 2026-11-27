import os
import random
import json
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..serialization import atomic_save, load_weights


def configure(seed, cfg):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    deterministic = cfg["deterministic"]
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.set_num_threads(cfg["cpu_threads"])
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else cfg["device"])


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class Training:
    def __init__(self, config, seed, device, progress_dir=None):
        self.config = config
        self.cfg = config["training"]
        self.seed = seed
        self.device = device
        self.generator = torch.Generator().manual_seed(seed)
        self.progress_dir = progress_dir

    def loader(self, dataset, shuffle=False):
        return DataLoader(dataset, batch_size=self.cfg["batch_size"], shuffle=shuffle,
                          num_workers=self.cfg["workers"], worker_init_fn=seed_worker,
                          generator=self.generator, pin_memory=self.device.type == "cuda")

    def move(self, batch):
        return [x.to(self.device, non_blocking=True) for x in batch]

    def adam(self, parameters, lr=None):
        return torch.optim.Adam(parameters, lr=lr or self.cfg["lr"]["initial"], weight_decay=self.cfg["weight_decay"])

    def step(self, loss, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(parameters, self.cfg["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        return loss.detach()

    def balanced(self, datasets):
        loaders = {station: self.loader(ds, shuffle=True) for station, ds in datasets.items()}
        return loaders, self.cfg["source_batches_per_epoch"] or max(map(len, loaders.values()))

    @torch.no_grad()
    def validation(self, model, datasets, residuals=None):
        model.eval()
        errors = []
        for dataset in datasets:
            total = torch.zeros((), device=self.device)
            count = 0
            for batch in self.loader(dataset):
                x, y, *context = self.move(batch)
                pred = model(x, residuals).power if residuals is not None else model(x, *context).power
                total += F.mse_loss(pred, y, reduction="sum")
                count += y.numel()
            errors.append(float(total / count))
        return float(np.mean(errors))

    def early_fit(self, epochs, tracked, train_epoch, validation, optimizer=None, phase=None):
        best, best_states, stale, history = float("inf"), None, 0, []
        progress = self.progress_dir/f"{phase}.pt" if phase and self.progress_dir else None
        if progress and progress.exists():
            state = load_weights(progress)
            best, best_states, stale, history = (state[k] for k in ("best", "best_states", "stale", "history"))
            for module, weights in zip(tracked, state["current"]):
                module.load_state_dict(weights)
            optimizer.load_state_dict(state["optimizer"])
            self.restore_rng(state["rng"])
        for epoch in range(len(history), epochs):
            if stale >= self.cfg["patience"]:
                break
            epoch_started = perf_counter()
            report = train_epoch(epoch)
            score = validation()
            if not np.isfinite(score):
                raise ValueError("Validation loss is nonfinite.")
            entry = {"epoch": epoch + 1, "validation_mse": score,
                     "epoch_seconds": perf_counter()-epoch_started, **report}
            entry = {key: None if isinstance(value, float) and not np.isfinite(value) else value
                     for key, value in entry.items()}
            history.append(entry)
            if self.progress_dir:
                self.progress_dir.mkdir(parents=True, exist_ok=True)
                with (self.progress_dir.parent/"training.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"phase": phase, **entry}, allow_nan=False) + "\n")
            if score < best:
                best, stale = score, 0
                best_states = [{k: v.detach().cpu().clone() for k, v in model.state_dict().items()} for model in tracked]
            else:
                stale += 1
            if progress:
                atomic_save({"best": best, "best_states": best_states, "stale": stale, "history": history,
                             "current": [module.state_dict() for module in tracked],
                             "optimizer": optimizer.state_dict(), "rng": self.rng_state()}, progress)
        if best_states is None:
            raise ValueError("Each active training phase requires at least one epoch.")
        for model, state in zip(tracked, best_states):
            model.load_state_dict(state)
        return history

    def rng_state(self):
        numpy_state = np.random.get_state()
        return {"torch": torch.get_rng_state(), "loader": self.generator.get_state(),
                "python": random.getstate(),
                "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []}

    def restore_rng(self, state):
        torch.set_rng_state(state["torch"])
        self.generator.set_state(state["loader"])
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            kind, keys, position, gaussian, cached = state["numpy"]
            np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
        if self.device.type == "cuda" and state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])


class RepeatBatches:
    """Reshuffle on exhaustion without storing completed batches."""
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def fit_prediction(tr, model, training, validation, epochs, optimizer=None, predict=None, penalty=None, tracked=None,
                   step_limit=None, phase=None):
    optimizer = optimizer or tr.adam(model.parameters())
    predict = predict or (lambda x, *context: model(x, *context).power)
    loaders = [tr.loader(ds, shuffle=True) for ds in training]
    num_steps = step_limit or max(map(len, loaders))

    def train_epoch(epoch):
        model.train()
        iterators = [RepeatBatches(loader) for loader in loaders]
        losses = []
        for _ in range(num_steps):
            batches = [tr.move(iterator.next()) for iterator in iterators]
            joined = [torch.cat(column) for column in zip(*batches)]
            outputs = predict(joined[0], *joined[2:]).split([len(batch[1]) for batch in batches])
            terms = [F.mse_loss(yhat, batch[1]) for yhat, batch in zip(outputs, batches)]
            loss = torch.stack(terms).mean()
            if penalty is not None:
                loss = loss + penalty()
            losses.append(tr.step(loss, optimizer))
        return {"train_loss": float(torch.stack(losses).mean())}

    return tr.early_fit(epochs, tracked or [model], train_epoch, validation, optimizer, phase)


def param_count(module):
    return sum(p.numel() for p in module.parameters())
