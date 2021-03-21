import os
import torch
from torch import nn
import torch.nn.functional as F

import models
import data
from dynamic_convolutions import DynamicConvolution, FlexibleKernelsDynamicConvolution
from train import test
from utils.options import Options


def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)


attentions_register = {}


class InspectDynamicConvolution(DynamicConvolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, temperature=1):
        batch_size = x.shape[0]

        alphas = self.attention(x, temperature)
        self.register(x, alphas)

        agg_weights = torch.sum(
            torch.mul(self.kernels_weights.unsqueeze(0), alphas.view(batch_size, -1, 1, 1, 1, 1)), dim=1)
        # Group the weights for each batch to conv2 all at once
        agg_weights = agg_weights.view(-1,
                                       *agg_weights.shape[-3:])  # batch_size*out_c X in_c X kernel_size X kernel_size
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

    def register(self, x, attention):
        resolution = x.shape[-2:]
        key = "x".join(map(str, resolution))
        if key not in attentions_register:
            attentions_register[key] = []
        attentions_register[key].append(attention.detach().cpu().clone())


def inspect_convolution_generator(nof_kernels, reduce):
    return FlexibleKernelsDynamicConvolution(InspectDynamicConvolution, nof_kernels, reduce)


def compute_entropy(prob_dict):
    def one_resolution(prob):
        cat = torch.cat(prob)
        return torch.distributions.Categorical(cat).entropy().mean()
    return {k: one_resolution(prob) for k, prob in prob_dict.items()}


def get_inspect_model(opt: Options):
    existing_dict = torch.load(opt.checkpoint_path, map_location=opt.device)['model']

    model_class = getattr(models, opt.model_class)
    conv_layer = inspect_convolution_generator(opt.nof_kernels, opt.reduce)
    model = model_class(conv_layer, *opt.model_extra_args).to(opt.device)
    model.load_state_dict(existing_dict)
    return model


def main():
    experiment_name = "Resnet_DY_05"
    epoch = 100

    experiment_path = os.path.join("experiments", experiment_name)
    opt = Options(suppress_parse=True)
    opt.load_from_file(os.path.join(experiment_path, "config.yaml"))
    opt.checkpoint_path = os.path.join(experiment_path, f"{experiment_name}_{epoch}.pth")
    assert opt.use_dynamic

    model = get_inspect_model(opt)

    test_dl = data.create_data_loader(opt, "test")
    test_score = test(model, opt.temperature[1], test_dl, opt.device)

    entropy = compute_entropy(attentions_register)

    print(test_score)
    print(entropy)


if __name__ == '__main__':
    main()
