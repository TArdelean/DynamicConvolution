import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..deeplab_details.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from dynamic_convolutions import DynamicConvolution, TempModule
from models.common import BaseModel, CustomSequential

class Decoder(TempModule):
    def __init__(self, num_classes, backbone, BatchNorm, ConvLayer, wm=1.0):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = ConvLayer(low_level_inplanes, int(48*wm), 1, bias=False)
        self.bn1 = BatchNorm(int(48*wm))
        # self.conv1 = ConvLayer(int(low_level_inplanes*wm), int(48*wm), 1, bias=False)
        # self.bn1 = BatchNorm(int(48*wm))
        self.relu = nn.ReLU()
        self.last_conv = CustomSequential(ConvLayer(int(304*wm), int(256*wm), kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(int(256*wm)),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       ConvLayer(int(256*wm), int(256*wm), kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(int(256*wm)),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       ConvLayer(int(256*wm), num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat, temperature):
        low_level_feat = self.conv1(low_level_feat, temperature)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x, temperature)

        return x

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

def build_decoder(num_classes, backbone, BatchNorm, ConvLayer, wm=1.0):
    return Decoder(num_classes, backbone, BatchNorm, ConvLayer, wm=wm)