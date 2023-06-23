import torch
import torchvision
from torch import nn
import numpy as np
import os

# MICRO RESNET
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        
        self.resblock = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
        )
        
    def forward(self, x):
        out = self.resblock(x)
        return out + x
    
class Upsample2d(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample2d, self).__init__()
        
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode='nearest')
        return x

class MicroResNet(nn.Module):
    def __init__(self):
        super(MicroResNet, self).__init__()
        
        self.downsampler = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 8, kernel_size=9, stride=4),
            nn.InstanceNorm2d(8, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
        )
        
        self.residual = nn.Sequential(
            ResBlock(32),
            nn.Conv2d(32, 64, kernel_size=1, bias=False, groups=32),
            ResBlock(64),
        )
        
        self.segmentator = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 16, kernel_size=3),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(),
            Upsample2d(scale_factor=2),
            nn.ReflectionPad2d(4),
            nn.Conv2d(16, 1, kernel_size=9),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.downsampler(x)
        out = self.residual(out)
        out = self.segmentator(out)
        return out

# SmartCrop module
class SmartCrop(nn.Module):
    def __init__(self, output_size, randomize_p=0.0, randomize_q=0.1, temperature=0.03):
        super().__init__()
        self.output_size = output_size
        self.randomize_p, self.randomize_q = randomize_p, randomize_q
        self.temperature = temperature
        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)
        self.saliency_model = MicroResNet().eval().requires_grad_(False)
        checkpoint = torch.load(os.path.dirname(__file__) + "/models/saliency_model_v9.pt", map_location="cpu")
        self.saliency_model.load_state_dict(checkpoint)

    def forward(self, image):
        is_batch = len(image.shape) == 4
        if not is_batch:
            image = image.unsqueeze(0)
        with torch.no_grad():
            resized_image = torchvision.transforms.functional.resize(image, 240, antialias=True)
            saliency_map = self.saliency_model(resized_image)
            tempered_heatmap = saliency_map.view(saliency_map.size(0), -1).div(self.temperature).softmax(-1)
            tempered_heatmap = tempered_heatmap / tempered_heatmap.sum(dim=1)
            tempered_heatmap = (tempered_heatmap > tempered_heatmap.max(dim=-1)[0]*0.75).float()
            saliency_map = tempered_heatmap.view(*saliency_map.shape)

        # GET CENTROID 
        coord_space = torch.cat([
            torch.linspace(0, 1, saliency_map.size(-2))[None, None, :, None].expand(-1, -1, -1, saliency_map.size(-1)),
            torch.linspace(0, 1, saliency_map.size(-1))[None, None, None, :].expand(-1, -1, saliency_map.size(-2), -1),
        ], dim=1)
        centroid = (coord_space * saliency_map).sum(dim=[-1, -2]) / saliency_map.sum(dim=[-1, -2])
        # CROP
        crops = []
        for i in range(image.size(0)):
            if np.random.rand() < self.randomize_p:
                centroid[i, 0] += np.random.uniform(-self.randomize_q, self.randomize_q)
                centroid[i, 1] += np.random.uniform(-self.randomize_q, self.randomize_q)
            top = (centroid[i, 0]*image.size(-2)-self.output_size[-2]/2).clamp(min=0, max=max(0, image.size(-2)-self.output_size[-2])).int()
            left = (centroid[i, 1]*image.size(-1)-self.output_size[-1]/2).clamp(min=0, max=max(0, image.size(-1)-self.output_size[-1])).int()
            bottom, right = top + self.output_size[-2], left + self.output_size[-1]
            crop = image[i, :, top:bottom, left:right]
            if crop.size(-2) < self.output_size[-2] or crop.size(-1) < self.output_size[-1]:
                crop = torchvision.transforms.functional.center_crop(crop, self.output_size)
            crops.append(crop)
        if is_batch:
            crops = torch.stack(crops, dim=0)
        else:
            crops = crops[0]
        return crops