import torch
from torch import nn

class _WeigthNorm(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps
        
    def _normalize(self, w):
        norm_dims = list(range(1, len(w.shape)))
        w_norm = torch.linalg.vector_norm(w, dim=norm_dims, keepdim=True)
        # w_norm = torch.norm_except_dim(w, 2, 0).clone()
        return w / (w_norm + self.eps)

    def forward(self, w):
        if self.training:
            with torch.no_grad():
                fan_in = w[0].numel()**0.5
                w.data = self._normalize(w.data.clone()) * fan_in
                # w.copy_(self._normalize(w) * fan_in)
        return self._normalize(w)

def apply_weight_norm(module, name="weight", init_weight=True): # this reparametrizes the parameters of a single module
    if init_weight:
        torch.nn.init.normal(getattr(module, name))
    nn.utils.parametrize.register_parametrization(module, name, _WeigthNorm(), unsafe=True)
    return module

def weight_norm_model(model, whitelist=None, init_weight=True):
    whitelist = whitelist or []

    def check_parameter(module, name):
        return hasattr(module, name) and not torch.nn.utils.parametrize.is_parametrized(module, name) and isinstance(getattr(module, name), nn.Parameter)

    for name, module in model.named_modules(): # this reparametrizes all layers of the model that have a "weight" parameter
        if not any([w in name for w in whitelist]):
            if check_parameter(module, "weight"):
                apply_weight_norm(module, init_weight=init_weight)
            elif check_parameter(module, "in_proj_weight"):
                apply_weight_norm(module, 'in_proj_weight', init_weight=init_weight)
    return model

def remove_weight_norm(model):
    for module in model.modules():
        if torch.nn.utils.parametrize.is_parametrized(module, "weight"):
            nn.utils.parametrize.remove_parametrizations(module, "weight")
        elif torch.nn.utils.parametrize.is_parametrized(module, "in_proj_weight"):
            nn.utils.parametrize.remove_parametrizations(module, "in_proj_weight")