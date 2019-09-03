import torch.nn as nn
import torch, math, sys

####
# CODE TAKEN FROM https://github.com/sdoria/SimpleSelfAttention
####

#Unmodified from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
def conv1d(ni, no, ks=1, stride=1, padding=0, bias=False):
	"Create and initialize a `nn.Conv1d` layer with spectral normalization."
	conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
	nn.init.kaiming_normal_(conv.weight)
	if bias: conv.bias.data.zero_()
	return nn.utils.spectral_norm(conv)

# Adapted from SelfAttention layer at https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
# Inspired by https://arxiv.org/pdf/1805.08318.pdf
class SimpleSelfAttention(nn.Module):
	
	def __init__(self, n_in, ks=1, sym=False):
		super().__init__()        
		self.conv = conv1d(n_in, n_in, ks, padding=ks//2, bias=False)            
		self.gamma = nn.Parameter(torch.Tensor([0.]))      
		self.sym = sym
		self.n_in = n_in
		
	def forward(self, x):
		if self.sym:
			# symmetry hack by https://github.com/mgrankin
			c = self.conv.weight.view(self.n_in,self.n_in)
			c = (c + c.t())/2
			self.conv.weight = c.view(self.n_in,self.n_in,1)
				
		size = x.size()  
		x = x.view(*size[:2],-1)   # (C,N)
		
		# changed the order of mutiplication to avoid O(N^2) complexity
		# (x*xT)*(W*x) instead of (x*(xT*(W*x)))
		
		convx = self.conv(x)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
		xxT = torch.bmm(x, x.permute(0,2,1).contiguous())   # (C,N) * (N,C) = (C,C)   => O(NC^2)  		    
		o = torch.bmm(xxT, convx)   # (C,C) * (C,N) = (C,N)   => O(NC^2)         
		o = self.gamma * o + x  
		  
		return o.view(*size).contiguous()  