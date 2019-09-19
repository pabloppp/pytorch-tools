import torch

def total_variation(X, reduction='sum'):
	tv_h = torch.abs(X[:, :, :, 1:] - X[:, :, :, :-1])
	tv_v = torch.abs(X[:, :, 1:] - X[:, :, :-1])

	tv = torch.mean(tv_h) + torch.mean(tv_v) if reduction == 'mean' else torch.sum(tv_h) + torch.sum(tv_v)
	
	return tv