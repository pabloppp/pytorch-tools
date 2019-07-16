import torch
from torch import nn
from functional.vq import vector_quantization

class VectorQuantize(nn.Module):
	def __init__(self, embedding_size, k):
		"""
		Takes an input of size (batch, embedding_size).
		Returns two lists of the nearest neigbour embeddings to each of the inputs, 
		with size (batch, embedding_size).
		The first list doesn't perform grads, the second one does.
		"""
		super(VectorQuantize, self).__init__()

		self.codebook = nn.Embedding(k, embedding_size)
		self.codebook.weight.data.uniform_(-1./k, 1./k)	
		self.vq = vector_quantization.apply

	def forward(self, z_e_x):
		z_q_x, indices = self.vq(z_e_x, self.codebook.weight.detach())
		z_q_x_grd = torch.index_select(self.codebook.weight, dim=0, index=indices)
		return z_q_x, z_q_x_grd