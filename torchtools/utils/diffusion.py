import torch

# Custom simplified foward/backward diffusion (cosine schedule)
class Diffuzz():
    def __init__(self, s=0.008, device="cpu"):
        self.s = torch.tensor([s]).to(device)
        self._init_alpha_cumprod = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

    def _alpha_cumprod(self, t):
        alpha_cumprod = torch.cos((t + self.s) / (1 + self.s) * torch.pi * 0.5) ** 2 / self._init_alpha_cumprod
        return alpha_cumprod.clamp(0.0001, 0.9999)

    def diffuse(self, x, t, noise=None): # t -> [0, 1]
        if noise is None:
            noise = torch.randn_like(x)
        alpha_cumprod = self._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        return alpha_cumprod.sqrt() * x + (1-alpha_cumprod).sqrt() * noise, noise

    def undiffuse(self, x, t, t_prev, noise, mode='ddpm'):
        alpha_cumprod = self._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        alpha_cumprod_prev = self._alpha_cumprod(t_prev).view(t_prev.size(0), *[1 for _ in x.shape[1:]])
        alpha = (alpha_cumprod / alpha_cumprod_prev)
                
        if mode == 'ddpm':
            mu = (1.0 / alpha).sqrt() * (x - (1-alpha) * noise / (1-alpha_cumprod).sqrt())
            std = ((1-alpha) * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)).sqrt() * torch.randn_like(mu)
            return mu + std * (t_prev != 0).float().view(t_prev.size(0), *[1 for _ in x.shape[1:]])
        elif mode == 'ddim':
            x0 = (x - (1 - alpha_cumprod).sqrt() * noise) / (alpha_cumprod).sqrt()
            dp_xt = (1 - alpha_cumprod_prev).sqrt()
            return (alpha_cumprod_prev).sqrt() * x0 + dp_xt * noise
        
    def p2_weight(self, t, k=1.0, gamma=1.0):
        alpha_cumprod = self._alpha_cumprod(t)
        return (k + alpha_cumprod / (1 - alpha_cumprod)) ** -gamma