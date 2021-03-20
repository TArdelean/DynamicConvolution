
# ALL ACKNOWLEDGMENT GOES TO THE PAPER & REPOSITORY AUTHORS
# https://github.com/jfzhang95/pytorch-deeplab-xception

import torch
import torch.nn as nn
import torch.nn.functional as F
from .deeplab_details.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .deeplab_details.aspp import build_aspp
from .deeplab_details.decoder import build_decoder
from .deeplab_details.backbone import build_backbone

#from .. import dynamic_convolutions 
#from dynamic_convolutions import DynamicConvolution, TempModule
from models.common import BaseModel, CustomSequential

# DeepLabV3+ model from paper "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (2018)
# https://paperswithcode.com/paper/encoder-decoder-with-atrous-separable

# With their model they achieved 89% of mean intersection-over-union score
# on PascalVOC-2012, which makes it second best model as of now (the best model at
# the moment is based on enormous amount of self training, and also no source
# code available)

__all__ = ['DeepLab', 'deeplab']

class DeepLab(BaseModel):
    def __init__(self, ConvLayer, backbone='mobilenet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, lr=0.007):
        super().__init__(ConvLayer)
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self._lr = lr
        self.freeze_bn = freeze_bn

    def forward(self, input, temperature):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
    def parameters(self):
        return [{'params': self.get_1x_lr_params(), 'lr': self._lr},
                {'params': self.get_10x_lr_params(), 'lr': self._lr * 10}]
                
if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


