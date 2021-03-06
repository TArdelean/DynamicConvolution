import os
import torch

from utils.options import Options


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    if path:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
    else:
        epoch = 0
    if optimizer:
        return model, epoch, optimizer
    else:
        return model, epoch


def save_checkpoint(model, optimizer, epoch, opt: Options):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }, os.path.join(opt.checkpoints_dir, f"{opt.experiment_name}_{epoch}.pth"))
