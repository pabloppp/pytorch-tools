import torch
from torch import nn
from .functional.vq import vector_quantize, binarize

class VectorQuantize(nn.Module):
	def __init__(self, embedding_size, k, ema_decay=0.99, ema_loss=False):
		"""
		Takes an input of size (batch, embedding_size).
		Returns two lists of the nearest neigbour embeddings to each of the inputs, 
		with size (batch, embedding_size).
		The first list doesn't perform grads, the second one does.
		"""
		super(VectorQuantize, self).__init__()

		self.codebook = nn.Embedding(k, embedding_size)
		self.codebook.weight.data.uniform_(-1./k, 1./k)	
		self.vq = vector_quantize.apply

		self.ema_decay = ema_decay
		self.ema_loss = ema_loss
		if ema_loss:
			self.register_buffer('ema_element_count', torch.ones(k))
			self.register_buffer('ema_weight_sum', torch.zeros_like(self.codebook.weight))		

	def _laplace_smoothing(self, x, epsilon):
		n = torch.sum(x)
		return ((x + epsilon) / (n + x.size(0) * epsilon) * n)

	def _updateEMA(self, z_e_x, z_q_x):
		expanded = self.codebook.weight.unsqueeze(-2).expand(*self.codebook.weight.shape[:-1], z_q_x.size(0), -1)
		mask = (expanded == z_q_x).float()
		elem_count = mask.mean(dim=-1).sum(dim=-1)
		weight_sum = (mask * z_e_x).sum(-2)
		
		self.ema_element_count = (self.ema_decay * self.ema_element_count) + ((1-self.ema_decay) * elem_count)
		self.ema_element_count = self._laplace_smoothing(self.ema_element_count, 1e-5)		
		self.ema_weight_sum = (self.ema_decay * self.ema_weight_sum) + ((1-self.ema_decay) * weight_sum)
		
		self.codebook.weight.data = self.ema_weight_sum / (self.ema_element_count.unsqueeze(-1))

	def forward(self, z_e_x):
		z_q_x, indices = self.vq(z_e_x, self.codebook.weight.detach())
		z_q_x_grd = torch.index_select(self.codebook.weight, dim=0, index=indices)
		if self.ema_loss and self.training:
			self._updateEMA(z_e_x.detach(), z_q_x.detach())
		return z_q_x, z_q_x_grd

class Binarize(nn.Module):
	def __init__(self, threshold=0.5):
		"""
		Takes an input of any size.
		Returns an output of the same size but with its values binarized (0 if input is below a threshold, 1 if its above)
		"""
		super(Binarize, self).__init__()

		self.bin = binarize.apply
		self.threshold = threshold

	def forward(self, x):
		return self.bin(x, self.threshold)