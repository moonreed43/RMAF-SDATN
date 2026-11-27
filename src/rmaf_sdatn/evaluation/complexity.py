from time import perf_counter

import torch


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def complexity(model, training_params, training_time, peak, example, tr):
    cfg, e = tr.config["model"], tr.config["evaluation"]
    x = torch.as_tensor(example[:1], device=tr.device)
    if model is not None:
        model.eval()

    def call():
        if model is None:
            return x[:, -1, :].expand(-1, cfg["horizon"])
        from ..models.cyclenet import CycleNet
        if isinstance(model, CycleNet):
            return model(x, torch.zeros(1, dtype=torch.long, device=tr.device))
        return model(x)

    for _ in range(e["latency_warmup"]):
        call()
    sync(tr.device)
    start = perf_counter()
    for _ in range(e["latency_repeats"]):
        call()
        sync(tr.device)
    elapsed = (perf_counter() - start) * 1000 / e["latency_repeats"]
    deployed = sum(p.numel() for p in model.parameters()) if model is not None else 0
    # Matrix multiply/add FLOPs using 2 per multiply-accumulate. Excludes
    # nonlinearities, bias, gate arithmetic and attention normalization.
    flops = None
    if model is None:
        flops = 0
    elif hasattr(model, "backbone"):
        h, t = cfg["hidden"], cfg["history"]
        flops = 2*t*4*h*(1+h+(cfg["layers"]-1)*2*h)
        if hasattr(model, "evolution"):
            s, sh, sup, emb = len(cfg["scales"]), cfg["shared_dim"], cfg["supplementary_dim"], cfg["embedding"]
            flops += 2*(t*((h*(s+1)+s)*emb+emb+h*sh)+h*sup+(sh+sup)*cfg["horizon"])
        else:
            flops += 2*h*cfg["horizon"]
    else:
        # The profiler reports FLOPs for supported matrix and convolution operators.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as trace:
            call()
            sync(tr.device)
        flops = sum(event.flops for event in trace.key_averages())
    return {"training_params": training_params, "deployment_params": deployed, "training_seconds": training_time,
            "single_sample_latency_ms": elapsed, "affine_flops": flops, "peak_train_cuda_bytes": peak,
            "device": str(tr.device), "gpu_name": torch.cuda.get_device_name(tr.device) if tr.device.type == "cuda" else None,
            "torch_version": str(torch.__version__)}
