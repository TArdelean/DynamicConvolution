"""
Code adapted from here: https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py
"""

import torch
import torch.nn as nn
import math

from models.common import BaseModel, CustomSequential
from dynamic_convolutions import DynamicConvolution, TempModule, dynamic_convolution_generator


__all__ = ['mobilenetv2', 'MobileNetV2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride, conv=nn.Conv2d):
    return CustomSequential(
        conv(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv=nn.Conv2d):
    return CustomSequential(
        conv(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(TempModule):
    def __init__(self, inp, oup, stride, expand_ratio, conv=nn.Conv2d):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = CustomSequential(
                # dw
                conv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = CustomSequential(
                # pw
                conv(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                conv(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                conv(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x, temperature):
        if self.identity:
            return x + self.conv(x, temperature)
        else:
            return self.conv(x, temperature)


class MobileNetV2(BaseModel):
    def __init__(self, conv, num_classes=200, depth_multiplier=0.35):
        super(MobileNetV2, self).__init__(conv)
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * depth_multiplier, 4 if depth_multiplier == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2, conv=nn.Conv2d)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * depth_multiplier, 4 if depth_multiplier == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, conv=self.ConvLayer))
                input_channel = output_channel
        self.features = CustomSequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * depth_multiplier, 4 if depth_multiplier == 0.1 else 8) if depth_multiplier > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel, conv=self.ConvLayer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x, temperature):
        x = self.features(x, temperature)
        x = self.conv(x, temperature)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, DynamicConvolution):
                for i_kernel in range(m.nof_kernels):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.kernels_weights[i_kernel].data.normal_(0, math.sqrt(2. / n))
                if m.kernels_bias is not None:
                    m.kernels_bias.data.zeros_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)


if __name__ == '__main__':
    x = torch.rand(1, 3, 64, 64)
    conv_layer = dynamic_convolution_generator(4, 4)
    model = MobileNetV2(conv_layer)
    x = model(x, 30)
    print(x.size())