import torch
from torch import nn

class LearnableWeightedSum(nn.Module):
    def __init__(self):
        super().__init__()
        self.wa = nn.Parameter(torch.tensor([1.0]))
        self.wb = nn.Parameter(torch.tensor([1.0]))
        
    def forward(self, a, b):
        norm = (self.wa**2 + self.wb**2)**0.5
        return (self.wa*a + self.wb*b) / norm
    
class FixedWeightedSum(nn.Module):
    def __init__(self, t=0.3):
        super().__init__()
        self.register_buffer("t", torch.tensor([t]))
        
    def forward(self, a, b):
        norm = ((1-self.t)**2 + self.t**2)**0.5
        return ((1-self.t)*a + self.t*b) / norm