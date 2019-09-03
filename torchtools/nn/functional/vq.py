import torch
from torch.autograd import Function

class vector_quantize(Function):
	@staticmethod
	def forward(ctx, x, codebook):
		with torch.no_grad():
			codebook_sqr = torch.sum(codebook ** 2, dim=1)
			x_sqr = torch.sum(x ** 2, dim=1, keepdim=True)

			dist = torch.addmm(codebook_sqr + x_sqr, x, codebook.t(), alpha=-2.0, beta=1.0)
			_, indices = dist.min(dim=1)
			
			ctx.save_for_backward(indices, codebook)
			ctx.mark_non_differentiable(indices)

			nn = torch.index_select(codebook, 0, indices)
			return nn, indices
	
	@staticmethod
	def backward(ctx, grad_output, grad_indices):
		grad_inputs, grad_codebook = None, None
		
		if ctx.needs_input_grad[0]:
			grad_inputs = grad_output.clone()
		if ctx.needs_input_grad[1]:
			# Gradient wrt. the codebook
			indices, codebook = ctx.saved_tensors

			grad_codebook = torch.zeros_like(codebook)
			grad_codebook.index_add_(0, indices, grad_output)
		
		return (grad_inputs, grad_codebook)


class binarize(Function):
	@staticmethod
	def forward(ctx, x, threshold=0.5):
		with torch.no_grad():
			binarized = (x > threshold).float()
			ctx.mark_non_differentiable(binarized)

			return binarized
	
	@staticmethod
	def backward(ctx, grad_output):
		grad_inputs = None
		
		if ctx.needs_input_grad[0]:
			grad_inputs = grad_output.clone()
		
		return grad_inputs