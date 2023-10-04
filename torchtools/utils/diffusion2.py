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

# https://github.com/ozanciga/diffusion-for-beginners/blob/main/samplers/ddim.py
class DDIMSampler(SimpleSampler):
    def step(self, x, t, t_prev, pred, eta=0):
        alpha_cumprod = self.diffuzz._alpha_cumprod(t).view(t.size(0), *[1 for _ in x.shape[1:]])
        alpha_cumprod_prev = self.diffuzz._alpha_cumprod(t_prev).view(t_prev.size(0), *[1 for _ in x.shape[1:]])

        sigma_tau = eta * ((1 - alpha_cumprod_prev) / (1 - alpha_cumprod)).sqrt() * (1 - alpha_cumprod / alpha_cumprod_prev).sqrt() if eta > 0 else 0
        if self.mode == 'v':
            x0 = alpha_cumprod.sqrt() * x - (1-alpha_cumprod).sqrt() * pred
            noise = (1-alpha_cumprod).sqrt() * x + alpha_cumprod.sqrt() * pred
        elif self.mode == 'x':
            x0 = pred
            noise = (x - x0 * alpha_cumprod.sqrt()) / (1 - alpha_cumprod).sqrt()
        else:
            noise = pred
            x0 = (x - (1 - alpha_cumprod).sqrt() * noise) / alpha_cumprod.sqrt()
        renoised = alpha_cumprod_prev.sqrt() * x0 + (1 - alpha_cumprod_prev - sigma_tau ** 2).sqrt() * noise + sigma_tau * torch.randn_like(x)
        return x0, renoised, pred

class DDPMSampler(DDIMSampler):
    def step(self, x, t, t_prev, pred, eta=1):
        return super().step(x, t, t_prev, pred, eta)

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
            if self.scaler != 1:
                alpha_cumprod = (alpha_cumprod/(1-alpha_cumprod)).log().add(self.scaler).sigmoid().clamp(self.clamp_range[0], self.clamp_range[1])
            return alpha_cumprod
        else:
            return self.cached_steps[t.mul(len(self.cached_steps)-1).long()]

    def scale_t(self, t, scaler):
        scaler = 2 * np.log(1/scaler)
        alpha_cumprod = torch.cos((t + self.s) / (1 + self.s) * torch.pi * 0.5).clamp(0, 1) ** 2 / self._init_alpha_cumprod
        alpha_cumprod = alpha_cumprod.clamp(self.clamp_range[0], self.clamp_range[1])
        if scaler != 1:
            alpha_cumprod = (alpha_cumprod/(1-alpha_cumprod)).log().add(scaler).sigmoid().clamp(self.clamp_range[0], self.clamp_range[1])
        return (((alpha_cumprod * self._init_alpha_cumprod) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + self.s) - self.s

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
        return alpha_cumprod.sqrt() * noised - (1-alpha_cumprod).sqrt() * pred_v

    def noise_from_v(self, noised, pred_v, t):
        alpha_cumprod = self._alpha_cumprod(t).view(t.size(0), *[1 for _ in noised.shape[1:]])
        return (1-alpha_cumprod).sqrt() * noised + alpha_cumprod.sqrt() * pred_v

    def undiffuse(self, x, t, t_prev, pred, sampler=None, **kwargs):
        if sampler is None:
            sampler = DDPMSampler(self)
        return sampler(x, t, t_prev, pred, **kwargs)

    def sample(self, model, model_inputs, shape, mask=None, t_start=1.0, t_end=0.0, timesteps=20, x_init=None, cfg=3.0, cfg_rho=0.7, unconditional_inputs=None, sampler='ddpm', dtype=None, sample_mode='v', sampler_params={}, t_scaler=1):
        r_range = torch.linspace(t_start, t_end, timesteps+1)[:, None].expand(-1, shape[0] if x_init is None else x_init.size(0)).to(self.device)            
        if t_scaler != 1:
            r_range = self.scale_t(r_range, t_scaler)
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
        if cfg is not None:
            if unconditional_inputs is None:
                unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
            model_inputs = {k:torch.cat([v, v_u]) if isinstance(v, torch.Tensor) else None for (k, v), (k_u, v_u) in zip(model_inputs.items(), unconditional_inputs.items())}
        for i in range(0, timesteps):
            if mask is not None and x_init is not None:
                x_renoised, _ = self.diffuse(x_init, r_range[i])
                x = x * mask + x_renoised * (1-mask)
            if cfg is not None:
                pred, pred_unconditional = model(torch.cat([x] * 2), torch.cat([r_range[i]] * 2), **model_inputs).chunk(2)
                pred_cfg = torch.lerp(pred_unconditional, pred, cfg)
                if cfg_rho > 0:
                    std_pos, std_cfg = pred.std(),  pred_cfg.std()
                    pred = cfg_rho * (pred_cfg * std_pos/(std_cfg+1e-9)) + pred_cfg * (1-cfg_rho)
                else:
                    pred = pred_cfg
            else:
                pred = model(x, r_range[i], **model_inputs)

            diff_out = self.undiffuse(x, r_range[i], r_range[i+1], pred, sampler=sampler, **sampler_params)
            x = diff_out[1]
            altered_vars = yield diff_out
            
            # Update some running variables if the user wants
            if altered_vars is not None:
                cfg = altered_vars.get('cfg', cfg)
                cfg_rho = altered_vars.get('cfg_rho', cfg_rho)
                sampler = altered_vars.get('sampler', sampler)
                unconditional_inputs = altered_vars.get('unconditional_inputs', unconditional_inputs)
                model_inputs = altered_vars.get('model_inputs', model_inputs)
                x = altered_vars.get('x', x)
                mask = altered_vars.get('mask', mask)
                x_init = altered_vars.get('x_init', x_init)
        
    def p2_weight(self, t, k=1.0, gamma=1.0):
        alpha_cumprod = self._alpha_cumprod(t)
        return (k + alpha_cumprod / (1 - alpha_cumprod)) ** -gamma
    
    def truncated_snr_weight(self, t, min=1.0, max=None):
        alpha_cumprod = self._alpha_cumprod(t)
        srn = (alpha_cumprod / (1 - alpha_cumprod))
        if min != None or max != None:
            srn = srn.clamp(min=min, max=max)
        return srn
