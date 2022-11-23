from math import sqrt

import torch
from torch import nn


def reshape_text_features_to_concat(text_features, image_features_shapes):
    return text_features.view((*text_features.size(), 1, 1)).repeat(1, 1, *image_features_shapes[2:])


def calculate_mean_std(x):
    mu = torch.mean(x, dim=(2, 3), keepdim=True).detach()
    std = torch.std(x, dim=(2, 3), keepdim=True, unbiased=False).detach()
    return mu, std


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim, bias=bias)
        linear.weight.data.normal_()
        if bias:
            linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, inputs):
        return self.linear(inputs)