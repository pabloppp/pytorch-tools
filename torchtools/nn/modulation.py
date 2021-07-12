
import torch
from torch import nn
import math

####
# TOTALLY INSPIRED AND EVEN COPIED SOME CHUNKS FROM 
# https://github.com/rosinality/alias-free-gan-pytorch/blob/main/model.py#L143
# But made it extend from the base Conv2d to avoid some boilerplate
####
class ModulatedConv2d(nn.Conv2d):
    def __init__(self,  *args, demodulate=True, ema_decay=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        fan_in = self.in_channel * self.kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)

        self.demodulate = demodulate
        self.ema_decay = ema_decay
        self.register_buffer("ema_var", torch.tensor(1.0))

    def forward(self, x, w):
        batch, in_channel, height, width = x.shape

        style = w.view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.ema_decay < 1:
            if self.training:
                var = x.pow(2).mean((0, 1, 2, 3))
                self.ema_var.mul_(self.ema_decay).add_(var.detach(), alpha=1 - self.ema_decay)

            weight = weight / (torch.sqrt(self.ema_var) + 1e-8)

        input = x.view(1, batch * in_channel, height, width)
        self.groups = batch
        out = self._conv_forward(input, weight, self.bias)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out
