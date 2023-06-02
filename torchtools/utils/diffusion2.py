import torch
import numpy as np

# Samplers --------------------------------------------------------------------
class SimpleSampler():
    def __init__(self, diffuzz, mode="v"):
        self.current_step = -1
        self.diffuzz = diffuzz
        if mode not in ['v', 'e', 'x']:
            raise Exception("Mode should be either 'v', 'e' or 'x'")
        self.mode = mode

    def __call__(self, *args, **kwargs):
        self.current_step += 1
        return self.step(*args, **kwargs)

    def init_x(self, shape):
        return torch.randn(*shape, device=self.diffuzz.device)

    def step(self, x, t, t_prev, noise):
        raise NotImplementedError("You should override the 'apply' function.")

class DDPMSampler(SimpleSampler):
    def step(self, x, t, t_prev, pred):
        alpha_cumprod = self.diffuzz._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        alpha_cumprod_prev = self.diffuzz._alpha_cumprod(t_prev).view(t_prev.size(0), *[1 for _ in x.shape[1:]])

        sigma_tau = ((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)).sqrt() * (1 - alpha_cumprod / alpha_cumprod_prev).sqrt()
        if self.mode == 'v':
            x0 = alpha_cumprod * x - (1-alpha_cumprod).sqrt() * pred
            noise = (1-alpha_cumprod).sqrt() * x + alpha_cumprod * pred
        elif self.mode == 'x':
            x0 = pred
            noise = (x - x0 * (alpha_cumprod).sqrt()) / (1 - alpha_cumprod).sqrt()
        else:
            noise = pred
            x0 = (x - (1 - alpha_cumprod).sqrt() * noise) / (alpha_cumprod).sqrt()
        dp_xt = (1 - alpha_cumprod_prev).sqrt()
        return x0, (alpha_cumprod_prev).sqrt() * x0 + dp_xt * noise + sigma_tau * torch.randn_like(x), pred

# https://github.com/ozanciga/diffusion-for-beginners/blob/main/samplers/ddim.py
class DDIMSampler(SimpleSampler):
    def step(self, x, t, t_prev, pred):
        alpha_cumprod = self.diffuzz._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        alpha_cumprod_prev = self.diffuzz._alpha_cumprod(t_prev).view(t_prev.size(0), *[1 for _ in x.shape[1:]])

        if self.mode == 'v':
            x0 = alpha_cumprod * x - (1-alpha_cumprod).sqrt() * pred
            noise = (1-alpha_cumprod).sqrt() * x + alpha_cumprod * pred
        elif self.mode == 'x':
            x0 = pred
            noise = (x - x0 * (alpha_cumprod).sqrt()) / (1 - alpha_cumprod).sqrt()
        else:
            noise = pred
            x0 = (x - (1 - alpha_cumprod).sqrt() * noise) / (alpha_cumprod).sqrt()
        dp_xt = (1 - alpha_cumprod_prev).sqrt()
        return x0, (alpha_cumprod_prev).sqrt() * x0 + dp_xt * noise, pred

sampler_dict = {
    'ddpm': DDPMSampler,
    'ddim': DDIMSampler,
}

# Custom simplified foward/backward diffusion (cosine schedule)
class Diffuzz2():
    def __init__(self, s=0.008, device="cpu", cache_steps=None, scaler=1, clamp_range=(1e-7, 1-1e-7)):
        self.device = device
        self.s = torch.tensor([s]).to(device)
        self._init_alpha_cumprod = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2
        self.scaler = 2 * np.log(1/scaler)
        self.cached_steps = None
        self.clamp_range = clamp_range
        if cache_steps is not None:
            self.cached_steps = self._alpha_cumprod(torch.linspace(0, 1, cache_steps, device=device))

    def _alpha_cumprod(self, t):
        if self.cached_steps is None:
            alpha_cumprod = torch.cos((t + self.s) / (1 + self.s) * torch.pi * 0.5).clamp(0, 1) ** 2 / self._init_alpha_cumprod
            alpha_cumprod = alpha_cumprod.clamp(self.clamp_range[0], self.clamp_range[1])
            if self.scaler != 0:
                alpha_cumprod = (alpha_cumprod/(1-alpha_cumprod)).log().add(self.scaler).sigmoid().clamp(self.clamp_range[0], self.clamp_range[1])
            return alpha_cumprod
        else:
            return self.cached_steps[t.mul(len(self.cached_steps)-1).long()]

    def diffuse(self, x, t, noise=None): # t -> [0, 1]
        if noise is None:
            noise = torch.randn_like(x)
        alpha_cumprod = self._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        return alpha_cumprod.sqrt() * x + (1-alpha_cumprod).sqrt() * noise, noise
    
    def get_v(self, x, t, noise):
        alpha_cumprod = self._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        # x0 = alpha_cumprod * noised − (1-alpha_cumprod).sqrt() * pred_v
        # noise = (1-alpha_cumprod).sqrt() * noised + alpha_cumprod * pred_v
        return alpha_cumprod.sqrt() * noise - (1-alpha_cumprod).sqrt() * x
    
    def x0_from_v(self, noised, pred_v, t):
        alpha_cumprod = self._alpha_cumprod(t).view(t.size(0), *[1 for _ in noised.shape[1:]])
        return alpha_cumprod * noised - (1-alpha_cumprod).sqrt() * pred_v

    def noise_from_v(self, noised, pred_v, t):
        alpha_cumprod = self._alpha_cumprod(t).view(t.size(0), *[1 for _ in noised.shape[1:]])
        return (1-alpha_cumprod).sqrt() * noised + alpha_cumprod * pred_v

    def undiffuse(self, x, t, t_prev, pred, sampler=None):
        if sampler is None:
            sampler = DDPMSampler(self)
        return sampler(x, t, t_prev, pred)

    def sample(self, model, model_inputs, shape, mask=None, t_start=1.0, t_end=0.0, timesteps=20, x_init=None, cfg=3.0, cfg_rho=0.7, unconditional_inputs=None, sampler='ddpm', dtype=None, sample_mode='v'):
        r_range = torch.linspace(t_start, t_end, timesteps+1)[:, None].expand(-1, shape[0] if x_init is None else x_init.size(0)).to(self.device)            
        if isinstance(sampler, str):
            if sampler in sampler_dict:
                sampler = sampler_dict[sampler](self, sample_mode)
            else:
                raise ValueError(f"If sampler is a string it must be one of the supported samplers: {list(sampler_dict.keys())}")
        elif issubclass(sampler, SimpleSampler):
            sampler =  sampler(self, sample_mode)
        else:
            raise ValueError("Sampler should be either a string or a SimpleSampler object.")
    
        x = sampler.init_x(shape) if x_init is None or mask is not None else x_init.clone()
        if dtype is not None:
            r_range = r_range.to(dtype)
            x = x.to(dtype)
        for i in range(0, timesteps):
            if mask is not None and x_init is not None:
                x_renoised, _ = self.diffuse(x_init, r_range[i])
                x = x * mask + x_renoised * (1-mask)
            pred = model(x, r_range[i], **model_inputs)
            if cfg is not None:
                if unconditional_inputs is None:
                    unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
                pred_unconditional = model(x, r_range[i], **unconditional_inputs)
                pred_cfg = torch.lerp(pred_unconditional, pred, cfg)
                if cfg_rho > 0:
                    std_pos, std_cfg = pred.std(),  pred_cfg.std()
                    pred = cfg_rho * (pred_cfg * std_pos/(std_cfg+1e-9)) + pred_cfg * (1-cfg_rho)
                else:
                    pred = pred_cfg
            diff_out = self.undiffuse(x, r_range[i], r_range[i+1], pred, sampler=sampler)
            x = diff_out[1]
            altered_vars = yield diff_out
            
            # Update some running variables if the user wants
            if altered_vars is not None:
                cfg = altered_vars.get('cfg', cfg)
                cfg_rho = altered_vars.get('cfg_rho', cfg_rho)
                sampler = altered_vars.get('sampler', sampler)
                unconditional_inputs = altered_vars.get('unconditional_inputs', unconditional_inputs)
                model_inputs = altered_vars.get('model_inputs', model_inputs)
        
    def p2_weight(self, t, k=1.0, gamma=1.0):
        alpha_cumprod = self._alpha_cumprod(t)
        return (k + alpha_cumprod / (1 - alpha_cumprod)) ** -gamma
    
    def truncated_snr_weight(self, t, k=1.0):
         alpha_cumprod = self._alpha_cumprod(t)
         return (alpha_cumprod / (1 - alpha_cumprod)).clamp(min=k)