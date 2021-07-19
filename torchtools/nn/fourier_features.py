import torch
from torch import nn
import math

class FourierFeatures2d(nn.Module):
    def __init__(self, size, dim, cutoff, affine_eps=1e-8, freq_range=[-0.5, 0.5], w_scale=0, allow_scaling=False, op_order=['r', 't', 's']):
        super().__init__()
        self.size = size
        self.dim = dim
        self.cutoff = cutoff
        self.freq_range = freq_range
        self.affine_eps = affine_eps
        self.w_scale = w_scale
        coords = torch.linspace(freq_range[0], freq_range[1], size+1)[:-1]
        freqs = torch.linspace(0, cutoff, dim // 4)
        if w_scale > 0:
            freqs = freqs @ (torch.randn(dim // 4, dim // 4) * w_scale)
        coord_map = torch.outer(freqs, coords)
        coord_map = 2 * math.pi * coord_map
        self.register_buffer("coord_h", coord_map.view(freqs.shape[0], 1, size))
        self.register_buffer("coord_w", self.coord_h.transpose(1, 2).detach())
        self.register_buffer("lf", freqs.view(1, dim // 4, 1, 1) * 2*math.pi * 2/size)
        self.allow_scaling = allow_scaling
        for op in op_order:
            assert op in ['r', 't', 's'], f"Operation not valid: {op}"
        self.op_order = op_order

    def forward(self, affine):
        norm = ((affine[:, 0:1].pow(2) + affine[:, 1:2].pow(2)).sqrt() + self.affine_eps).expand(affine.size(0), 4)
        if self.allow_scaling:
            assert affine.size(-1) == 6, f"If scaling is enabled, 2 extra values must be passed for a total of 6, and not {affine.size(-1)}"
            norm = torch.cat([norm, norm.new_ones(affine.size(0), 2)], dim=1)
        else:
            assert affine.size(-1) == 4, f"If scaling is disabled, 4 affine values should be passed, and not {affine.size(-1)}"
        affine = affine / norm
        affine = affine[:, :, None, None, None]

        coord_h, coord_w = self.coord_h.unsqueeze(0), self.coord_w.unsqueeze(0)

        for op in reversed(self.op_order):
            if op == 's' and self.allow_scaling:
                coord_h = coord_h / nn.functional.threshold(affine[:, 5], 1.0, 1.0) # scale
                coord_w = coord_w / nn.functional.threshold(affine[:, 4], 1.0, 1.0)

            elif op == 't':
                coord_h = coord_h - (affine[:, 3] * self.lf) # shift
                coord_w = coord_w - (affine[:, 2] * self.lf) 
            
            elif op == 'r':
                _coord_h = -coord_w * affine[:, 1] + coord_h * affine[:, 0] # rotation
                coord_w = coord_w * affine[:, 0] + coord_h * affine[:, 1]
                coord_h = _coord_h

        coord_h = torch.cat((torch.sin(coord_h), torch.cos(coord_h)), 1)
        coord_w = torch.cat((torch.sin(coord_w), torch.cos(coord_w)), 1)

        coords = torch.cat((coord_h, coord_w), 1)
        return coords

    def extra_repr(self):
        info_string = f'size={self.size}, dim={self.dim}, cutoff={self.cutoff}, freq_range={self.freq_range}'
        if self.w_scale > 0:
            info_string += f', w_scale={self.w_scale}'
        if self.allow_scaling:
            info_string += f', allow_scaling={self.allow_scaling}'
        return info_string 
