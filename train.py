import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import models
from torch import nn
import data
from models import common
from models.common import TemperatureScheduler, SmoothNLLLoss
from utils.options import Options
from utils.segmentation_metrics import SegmentationEvaluator
import sys
import numpy as np
from functools import partial

from tqdm import tqdm

from utils.utils import load_checkpoint, save_checkpoint


def train_epoch(epoch: int, model: nn.Module, criterion: nn.Module, temperature: TemperatureScheduler,
                optimizer: torch.optim.Optimizer, loader: torch.utils.data.DataLoader, device: torch.device,
                writer: SummaryWriter = None, batch_average=False):
    model.train()
    print(f"Training epoch {epoch}")
    t_bar = tqdm(loader)
    iteration = (epoch-1) * len(loader.dataset)
    for b_idx, batch in enumerate(t_bar):
        in_data, target = batch
        in_data, target = in_data.to(device), target.to(device)
        out = model(in_data, temperature.get(epoch))
        loss = criterion(out, target)
        if batch_average:
            loss /= loader.batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iteration += len(in_data)
        t_bar.set_description(f"Current loss={loss.item():.3f}", refresh=True)
        if writer:
            writer.add_scalar("Loss/train", loss.item(), iteration)


# classification accuracy validation score
def test_accuracy(model: nn.Module, temperature: float, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            in_data, target = batch
            in_data, target = in_data.to(device), target.to(device)
            output = model(in_data, temperature)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()


    return correct / len(loader.dataset)

# segmentation validation scores
def test_segmentation(model: nn.Module, temperature: float, loader: torch.utils.data.DataLoader, device: torch.device, n_classes: int):
    model.eval()
    evaluator = SegmentationEvaluator(n_classes)
    with torch.no_grad():
        for batch in loader:
            in_data, target = batch
            in_data, target = in_data.to(device), target.to(device)
            with torch.no_grad():
                output = model(in_data, temperature)
            target = target.cpu().numpy()
            pred = output.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            evaluator.add_batch(target, pred)
    mIoU = evaluator.Mean_Intersection_over_Union()
    #fwIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    #px_accuracy = evaluator.Pixel_Accuracy()
    #px_accuracy_class = evaluator.Pixel_Accuracy_Class()
    return mIoU

def main(opt: Options):
    model = models.create_model(opt)
    print("Training with network:")
    print(model)
    writer = SummaryWriter(opt.checkpoints_dir)

    train_dl = data.create_data_loader(opt, "train")
    test_dl = data.create_data_loader(opt, "test")
    criterion = getattr(common, opt.criterion)(*opt.criterion_args)
    temperature = TemperatureScheduler(*opt.temperature)
    optimizer = getattr(torch.optim, opt.optimizer)(model.parameters(), *opt.optimizer_args)
    scheduler = getattr(torch.optim.lr_scheduler, opt.scheduler)(optimizer, *opt.scheduler_args)
    device = torch.device(opt.device)
    model, epoch, optimizer, scheduler = load_checkpoint(opt.checkpoint_path, model, optimizer, scheduler, device)

    if opt.is_classification:
        test_metric = test_accuracy
        metric_name = 'Accuracy'
    else: # segmentation
        test_metric = partial(test_segmentation, n_classes=opt.n_classes)
        metric_name = 'mIoU'

    print('Setting up complete, starting training')
    for ep in range(epoch + 1, opt.max_epoch+1):
        train_epoch(ep, model, criterion, temperature, optimizer, train_dl, device, writer, batch_average=opt.batch_average)
        test_score = test_metric(model, temperature.get(ep), test_dl, device)
        writer.add_scalar(f"{metric_name}/test", test_score, ep * len(test_dl.dataset))
        print(f"Test {metric_name} after {ep} epochs = {test_score}")
        scheduler.step()
        if ep % opt.save_freq == 0:
            save_checkpoint(model, optimizer, scheduler, ep, opt)


if __name__ == '__main__':
    options = Options(config_file_arg="config_path")
    main(options)
