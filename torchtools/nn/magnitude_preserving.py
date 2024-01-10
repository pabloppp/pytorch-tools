import torch
from torch import nn

class MP_GELU(nn.GELU):
    def forward(self, x):
        return super().forward(x) / 0.652 # ¯\_(ツ)_/¯

class MP_SiLU(nn.SiLU):
    def forward(self, x):
        return super().forward(x) / 0.596 # ¯\_(ツ)_/¯
    
class Gain(nn.Module):
    def __init__(self, init_w=0.0):
        super().__init__()
        self.g = nn.Parameter(torch.tensor([init_w]))

    def forward(self, x):
        return x * self.g

