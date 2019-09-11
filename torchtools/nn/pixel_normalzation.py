import torch
from torch import nn

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)