import sys

from utils.options import Options
from .common import Conv2dWrapper
from dynamic_convolutions import dynamic_convolution_generator
from .simple_conv import SimpleConvNet
from .mobilenetv3 import MobileNetV3
from .resnet import ResNet10


def create_model(opt: Options):
    current_module = sys.modules[__name__]
    model_class = getattr(current_module, opt.model_class)
    conv_layer = dynamic_convolution_generator(opt.nof_kernels, opt.reduce) if opt.use_dynamic else Conv2dWrapper
    return model_class(conv_layer, *opt.model_extra_args).to(opt.device)
