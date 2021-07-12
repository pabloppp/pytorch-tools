
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

        fan_in = self.in_channels * self.kernel_size[0] ** 2
        self.scale = 1 / math.sqrt(fan_in)

        self.demodulate = demodulate
        self.ema_decay = ema_decay
        self.register_buffer("ema_var", torch.tensor(1.0))
        nn.init.normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, w):
        batch, in_channels, height, width = x.shape

        style = w.view(batch, 1, in_channels, 1, 1)
        weight = self.scale * self.weight.unsqueeze(0) * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]
        )

        if self.ema_decay < 1:
            if self.training:
                var = x.pow(2).mean((0, 1, 2, 3))
                self.ema_var.mul_(self.ema_decay).add_(var.detach(), alpha=1 - self.ema_decay)

            weight = weight / (torch.sqrt(self.ema_var) + 1e-8)

        input = x.view(1, batch * in_channels, height, width)
        self.groups = batch
        out = self._conv_forward(input, weight, None)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channels, height, width)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out
