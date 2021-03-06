import sys

import torch.utils.data

from utils.options import Options
from .mnist_dataset import MNIST_dataset


def create_data_loader(opt: Options, stage):
    current_module = sys.modules[__name__]
    dataset_getter = getattr(current_module, opt.dataset_class)
    dataset = dataset_getter(stage)
    return torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                       shuffle=(stage == "train"))
