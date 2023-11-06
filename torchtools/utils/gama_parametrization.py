import torch
from torch import nn


class _GammaScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, w):
        return w * self.gamma

def apply_gamma_reparam(module, name="weight"): # this reparametrizes the parameters of a single module
    nn.utils.parametrizations.spectral_norm(module, name)
    nn.utils.parametrize.register_parametrization(module, name, _GammaScaling())
    return module

def gamma_reparam_model(model):
    for module in model.modules(): # this reparametrizes all linear layers of the model
        if isinstance(module, nn.Linear) and not torch.nn.utils.parametrize.is_parametrized(module, "weight"):
            apply_gamma_reparam(module, "weight")
        elif isinstance(module, nn.MultiheadAttention) and not torch.nn.utils.parametrize.is_parametrized(module, "in_proj_weight"):
            apply_gamma_reparam(module, "in_proj_weight")
    return model

def remove_gamma_parametrizations(model):
    for module in model.modules():
        if torch.nn.utils.parametrize.is_parametrized(module, "weight"):
            nn.utils.parametrize.remove_parametrizations(module, "weight")
        elif torch.nn.utils.parametrize.is_parametrized(module, "in_proj_weight"):
            nn.utils.parametrize.remove_parametrizations(module, "in_proj_weight")
