import os
import torch

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
