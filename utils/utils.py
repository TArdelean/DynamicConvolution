import os
import torch

import numpy as np
from utils.options import Options


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    if path:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler:
                scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"]
    else:
        epoch = 0
    if optimizer:
        if scheduler:
            return model, epoch, optimizer, scheduler
        else:
            return model, epoch, optimizer
    else:
        return model, epoch


def save_checkpoint(model, optimizer, scheduler, epoch, opt: Options):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }, os.path.join(opt.checkpoints_dir, f"{opt.experiment_name}_{epoch}.pth"))


def profile(model, inputs, repeats=1000):
    # Reference for counting flops: http://www.bnikolic.co.uk/blog/python/flops/2019/10/01/pytorch-count-flops.html
    from pypapi import papi_high
    from pypapi import events as papi_events

    papi_high.start_counters([
        papi_events.PAPI_SP_OPS,
    ])
    model.forward(*inputs)
    flops = papi_high.stop_counters()[0] / 1000000.0

    from time import perf_counter
    times = []
    for _ in range(repeats):
        t = perf_counter()
        model.forward(*inputs)
        times.append(perf_counter() - t)
    params = sum(p.numel() for p in model.parameters()) / 1000000.0
    times = np.array(times) * 1000
    return {"params(M)": params, "flops(M)": flops,
            "inf_time_mean(ms)": np.mean(times), "inf_time_std(ms)": np.std(times)}
