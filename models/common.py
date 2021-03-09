import torch
from torch import nn

from dynamic_convolutions import DynamicConvolution


class Conv2dWrapper(nn.Conv2d):
    """
    Wrapper for pytorch Conv2d class which can take additional parameters(like temperature) and ignores them.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


def dynamic_convolution_generator(nof_kernels, reduce):
    def conv_layer(*args, **kwargs):
        return DynamicConvolution(nof_kernels, reduce, *args, **kwargs)
    return conv_layer


class BaseModel(nn.Module):
    def __init__(self, ConvLayer):
        super().__init__()
        self.ConvLayer = ConvLayer


class TemperatureScheduler:
    def __init__(self, initial_value, final_value=None, final_epoch=None):
        self.initial_value = initial_value
        self.final_value = final_value if final_value else initial_value
        self.final_epoch = final_epoch if final_epoch else 1
        self.step = 0 if self.final_epoch == 1 else (final_value - initial_value) / (final_epoch - 1)

    def get(self, crt_epoch=None):
        crt_epoch = crt_epoch if crt_epoch else self.final_epoch
        return self.initial_value + (min(crt_epoch, self.final_epoch) - 1) * self.step
