import torch
import math
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(self, c_dim, hidden_dim, nof_kernels):
        super().__init__()
        self.global_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.to_scores = nn.Sequential(nn.Linear(c_dim, hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_dim, nof_kernels))

    def forward(self, x, temperature=1):
        out = self.global_pooling(x)
        scores = self.to_scores(out)
        return F.softmax(scores / temperature, dim=-1)


class DynamicConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nof_kernels, reduce=4, groups=1, bias=True, **kwargs):
        """
        Implementation of Dynamic convolution layer
        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param kernel_size: size of the kernel.
        :param groups: controls the connections between inputs and outputs.
        in_channels and out_channels must both be divisible by groups.
        :param nof_kernels: number of kernels to use.
        :param reduce: Refers to the size of the hidden layer in attention: hidden = in_channels // reduce
        :param bias: If True, convolutions also have a learnable bias
        """
        super().__init__()
        self.groups = groups
        self.conv_args = kwargs
        self.nof_kernels = nof_kernels
        self.attention = AttentionLayer(in_channels, in_channels // reduce, nof_kernels)
        kernel_size = _pair(kernel_size)
        self.kernels_weights = nn.Parameter(torch.Tensor(
            nof_kernels, out_channels, in_channels // groups, *kernel_size), requires_grad=True)
        if bias:
            self.kernels_bias = nn.Parameter(torch.Tensor(nof_kernels, out_channels), requires_grad=True)
        else:
            self.register_parameter('kernels_bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        for i_kernel in range(self.nof_kernels):
            init.kaiming_uniform_(self.kernels_weights[i_kernel], a=math.sqrt(5))
        if self.kernels_bias is not None:
            bound = 1 / math.sqrt(self.kernels_weights[0, 0].numel())
            nn.init.uniform_(self.kernels_bias, -bound, bound)

    def forward(self, x, temperature=1):
        batch_size = x.shape[0]

        alphas = self.attention(x, temperature)
        agg_weights = torch.sum(
            torch.mul(self.kernels_weights.unsqueeze(0), alphas.view(batch_size, -1, 1, 1, 1, 1)), dim=1)
        # Group the weights for each batch to conv2 all at once
        agg_weights = agg_weights.view(-1, *agg_weights.shape[-3:])  # batch_size*out_c X in_c X kernel_size X kernel_size
        if self.kernels_bias is not None:
            agg_bias = torch.sum(torch.mul(self.kernels_bias.unsqueeze(0), alphas.view(batch_size, -1, 1)), dim=1)
            agg_bias = agg_bias.view(-1)
        else:
            agg_bias = None
        x_grouped = x.view(1, -1, *x.shape[-2:])  # 1 X batch_size*out_c X H X W

        out = F.conv2d(x_grouped, agg_weights, agg_bias, groups=self.groups * batch_size,
                       **self.conv_args)  # 1 X batch_size*out_C X H' x W'
        out = out.view(batch_size, -1, *out.shape[-2:])  # batch_size X out_C X H' x W'

        return out


if __name__ == '__main__':
    torch.manual_seed(41)
    t = torch.randn(1, 3, 16, 16)
    conv = DynamicConvolution(3, 8, kernel_size=3, nof_kernels=3, reduce=1, padding=1, bias=True)
    print(conv(t, 10).sum())
