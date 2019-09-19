import torch
import torch.nn as nn
from .functional import total_variation

class TVLoss(nn.Module):
	def __init__(self, reduction='sum', alpha=1e-4):
		super(TVLoss, self).__init__()
		self.reduction = reduction
		self.alpha = alpha

	def forward(self, x):
		return total_variation(x, reduction=self.reduction) * self.alpha