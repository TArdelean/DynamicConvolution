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
