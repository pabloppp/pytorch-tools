import torch
from torch import nn

class AdaIN(nn.Module):
    def __init__(self, n_channels):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(n_channels)

    def forward(self, image, style):
        factor, bias = style.view(style.size(0), style.size(1), 1, 1).chunk(2, dim=1)
        result = self.norm(image) * factor + bias
        return result