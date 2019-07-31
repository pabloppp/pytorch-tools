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

	def _updateEMA(self, z_e_x, indices):
		mask = nn.functional.one_hot(indices, self.ema_element_count.size(0)).float()
		elem_count = mask.sum(dim=0)
		weight_sum = torch.mm(mask.t(), z_e_x)
		
		self.ema_element_count = (self.ema_decay * self.ema_element_count) + ((1-self.ema_decay) * elem_count)
		self.ema_element_count = self._laplace_smoothing(self.ema_element_count, 1e-5)		
		self.ema_weight_sum = (self.ema_decay * self.ema_weight_sum) + ((1-self.ema_decay) * weight_sum)
		
		self.codebook.weight.data = self.ema_weight_sum / (self.ema_element_count.unsqueeze(-1))

	def forward(self, x):
		z_e_x = x.view(-1, x.size(-1)) if len(x.shape) > 2 else x
		z_q_x, indices = self.vq(z_e_x, self.codebook.weight.detach())		
		if self.ema_loss and self.training:
			self._updateEMA(z_e_x.detach(), indices.detach())
		# pick the graded embeddings after updating the codebook in order to have a more accurate commitment loss
		z_q_x_grd = torch.index_select(self.codebook.weight, dim=0, index=indices)
		return z_q_x.view(x.shape), z_q_x_grd.view(x.shape)

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