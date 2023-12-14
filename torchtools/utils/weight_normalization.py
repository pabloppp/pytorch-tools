import torch
from torch import nn

# Karras Weight normalization
class _WeigthNorm(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps
        
    def _normalize(self, w):
        norm_dims = list(range(1, len(w.shape)))
        w_norm = torch.linalg.vector_norm(w.detach(), dim=norm_dims, keepdim=True)
        return w / (w_norm + self.eps) 

    def forward(self, w):
        if self.training:
            with torch.no_grad():
                fan_in = w[0].numel()**0.5
                w.copy_(self._normalize(w) * fan_in)
        return self._normalize(w)

def apply_weight_norm(module, name="weight"): # this reparametrizes the parameters of a single module
    nn.utils.parametrize.register_parametrization(module, name, _WeigthNorm())
    return module

def weight_norm_model(model):
    def check_parameter(module, name):
        return hasattr(module, name) and not torch.nn.utils.parametrize.is_parametrized(module, name) and isinstance(getattr(module, name), nn.Parameter)
    
    for module in model.modules(): # this reparametrizes all linear layers of the model
        if check_parameter(module, "weight"):
            apply_weight_norm(module)
        elif check_parameter(module, "in_proj_weight"):
            apply_weight_norm(module, 'in_proj_weight')
    return model

def remove_weight_norm(model):
    for module in model.modules():
        if torch.nn.utils.parametrize.is_parametrized(module, "weight"):
            nn.utils.parametrize.remove_parametrizations(module, "weight")
        elif torch.nn.utils.parametrize.is_parametrized(module, "in_proj_weight"):
            nn.utils.parametrize.remove_parametrizations(module, "in_proj_weight")