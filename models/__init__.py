import sys

from utils.options import Options
from .simple_conv import SimpleConvNet


def create_model(opt: Options):
    current_module = sys.modules[__name__]
    model_class = getattr(current_module, opt.model_class)
    return model_class()

