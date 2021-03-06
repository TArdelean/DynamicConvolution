import torch.utils.data

import models
from torch import nn
import data
from utils.options import opt
from tqdm import tqdm


def train_epoch(epoch: int, model: nn.Module, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, loader: torch.utils.data.DataLoader, device: torch.device):
    model.train()
    print(f"Training epoch {epoch}")
    t_bar = tqdm(loader)
    for b_idx, batch in enumerate(t_bar):
        in_data, target = batch
        in_data, target = in_data.to(device), target.to(device)
        out = model(in_data)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t_bar.set_description(f"Current loss={loss.item():.3f}", refresh=True)


def test(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in loader:
            in_data, target = batch
            in_data, target = in_data.to(device), target.to(device)
            output = model(in_data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    return correct / len(loader.dataset)


def main():
    model = models.create_model(opt)
    train_dl = data.create_data_loader(opt, "train")
    test_dl = data.create_data_loader(opt, "test")
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    device = torch.device(opt.device)
    for ep in range(1, 11):
        train_epoch(ep, model, criterion, optimizer, train_dl, device)
        print(f"Test accuracy after {ep} epochs = {test(model, test_dl, device)}")


if __name__ == '__main__':
    main()



