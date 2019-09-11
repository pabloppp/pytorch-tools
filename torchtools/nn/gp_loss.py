import torch
from torch import nn
from .functional import gradient_penalty

class GPLoss(nn.Module):
	def __init__(self, discriminator, l=10):
		super(GPLoss, self).__init__()
		self.discriminator = discriminator
		self.l = l

	def forward(self, real_data, fake_data):
		return gradient_penalty(self.discriminator, real_data, fake_data, self.l)