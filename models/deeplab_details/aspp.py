import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..deeplab_details.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from dynamic_convolutions import DynamicConvolution, TempModule
from models.common import BaseModel, CustomSequential

class _ASPPModule(TempModule):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm, ConvLayer):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = ConvLayer(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x, temperature):
        x = self.atrous_conv(x, temperature)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, DynamicConvolution):
                for i_kernel in range(m.nof_kernels):
                    nn.init.kaiming_normal_(m.kernels_weights[i_kernel], mode='fan_out')
                if m.kernels_bias is not None:
                    nn.init.zeros_(m.kernels_bias)

class ASPP(TempModule):
    def __init__(self, backbone, output_stride, BatchNorm, ConvLayer):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm, ConvLayer=ConvLayer)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm, ConvLayer=ConvLayer)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm, ConvLayer=ConvLayer)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm, ConvLayer=ConvLayer)

        self.global_avg_pool = CustomSequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             ConvLayer(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = ConvLayer(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x, temperature):
        x1 = self.aspp1(x, temperature)
        x2 = self.aspp2(x, temperature)
        x3 = self.aspp3(x, temperature)
        x4 = self.aspp4(x, temperature)
        x5 = self.global_avg_pool(x, temperature)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x, temperature)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, DynamicConvolution):
                for i_kernel in range(m.nof_kernels):
                    nn.init.kaiming_normal_(m.kernels_weights[i_kernel], mode='fan_out')
                if m.kernels_bias is not None:
                    nn.init.zeros_(m.kernels_bias)


def build_aspp(backbone, output_stride, BatchNorm, ConvLayer):
    return ASPP(backbone, output_stride, BatchNorm, ConvLayer)