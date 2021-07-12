
import torch
from torch import nn
import math

####
# TOTALLY INSPIRED AND EVEN COPIED SOME CHUNKS FROM 
# https://github.com/rosinality/alias-free-gan-pytorch/blob/main/stylegan2/model.py#L94
# But made it extend from the base modules to avoid some boilerplate
####

class EqualLinear(nn.Linear):
    def __init__(self, *args, bias_init=0, lr_mul=1, **kwargs):
        super().__init__(*args, **kwargs)

        self.scale = (1 / math.sqrt(self.in_features)) * lr_mul
        self.lr_mul = lr_mul

        nn.init.normal_(self.weight, std=1/lr_mul)
        if self.bias is not None:
            nn.init.constant_(self.bias, bias_init)

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.scale, self.bias * self.lr_mul)


class EqualConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        fan_in = self.in_channels * self.kernel_size[0] ** 2
        self.scale = 1 / math.sqrt(fan_in)

        nn.init.normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return self._conv_forward(x, self.weight * self.scale, self.bias)


class EqualLeakyReLU(nn.LeakyReLU):
    def __init__(self, *args, scale=2**0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale
    
    def forward(self, x):
        return super().forward(x) * self.scale