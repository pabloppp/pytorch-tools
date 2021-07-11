
import torch
from torch import nn
import math

####
# TOTALLY INSPIRED AND EVEN COPIED SOME CHUNKS FROM 
# https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py#L454
# But as a twist, I decided to make this a wrapper for a the regular convolution
####
class Modulated2d(nn.Module):
    def __init__(self, module, demod=True, eps = 1e-8, decay=1.0):
        super().__init__()
        self.demod = demod
        self.eps = eps
        self.module = module
        nn.init.kaiming_normal_(self.module.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.module.weight_orig = nn.Parameter(self.module.weight.data)
        del self.module._parameters['weight']
        if self.module.bias is not None:
            self.module.bias_orig = nn.Parameter(self.module.bias.data)
            del self.module._parameters['bias']
            self.module.bias = None
        else:
            self.module.bias_orig = None

        fan_in = self.module.weight_orig.in_channel * self.module.weight_orig.kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)

        self.register_buffer("ema", torch.tensor(1.0))
        self.decay = decay

    def forward(self, x, y):
        b, c, h, w = x.shape
        w1 = y[:, None, :, None, None]
        w2 = self.module.weight_orig[None, :, :, :, :]
        weights = self.scale * w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.module.weight_orig.size(0), *ws)

        if self.decay < 1:
            if self.training:
                var = x.pow(2).mean((0, 1, 2, 3))
                self.ema.mul_(self.decay).add_(var.detach(), alpha=1 - self.decay)
            weights = weights / (torch.sqrt(self.ema) + 1e-8)

        self.module.weight = weights
        if self.module.bias_orig is not None:
            self.module.bias = self.module.bias_orig.repeat(b)
        self.module.groups = b

        x = self.module(x)
        x = x.reshape(-1, self.module.weight_orig.size(0), h, w)
        return x