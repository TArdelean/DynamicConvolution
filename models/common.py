import torch
from torch import nn


class Conv2dWrapper(nn.Conv2d):
    """
    Wrapper for pytorch Conv2d class which can take additional parameters(like temperature) and ignores them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


class TempModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, temperature) -> torch.Tensor:
        return x


class BaseModel(TempModule):
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


class CustomSequential(TempModule):
    """Sequential container that supports passing temperature to TempModule"""

    def __init__(self, *args):
        super().__init__()
        self.layers = nn.ModuleList(args)

    def forward(self, x, temperature):
        for layer in self.layers:
            if isinstance(layer, TempModule):
                x = layer(x, temperature)
            else:
                x = layer(x)
        return x
