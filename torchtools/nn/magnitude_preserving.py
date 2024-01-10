import torch
from torch import nn

class MP_GELU(nn.GELU):
    def forward(self, x):
        return super().forward(x) / 0.652 # ¯\_(ツ)_/¯

class MP_SiLU(nn.SiLU):
    def forward(self, x):
        return super().forward(x) / 0.596 # ¯\_(ツ)_/¯
    
class Gain(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        return x * self.g

class PixelNorm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
 
    def forward(self, x):
        return x / (torch.sqrt(torch.mean(x ** 2, dim=self.dim, keepdim=True)) + 1e-4)
