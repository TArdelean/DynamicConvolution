import sys

from utils.options import Options
from .common import dynamic_convolution_generator, Conv2dWrapper
from .simple_conv import SimpleConvNet


def create_model(opt: Options):
    current_module = sys.modules[__name__]
    model_class = getattr(current_module, opt.model_class)
    conv_layer = dynamic_convolution_generator(opt.nof_kernels, opt.reduce) if opt.use_dynamic else Conv2dWrapper
    return model_class(conv_layer).to(opt.device)
