import torch
from torch import nn
import math
from .stylegan2 import upfirdn2d

####
# TOTALLY INSPIRED AND EVEN COPIED SOME CHUNKS FROM 
# https://github.com/rosinality/alias-free-gan-pytorch/blob/main/model.py#L225
# But I simplified it into a single (almots) self-contained module. 
# Probably I give this module too much reponsibility but meh...
####

class AliasFreeActivation(nn.Module):
    def __init__(self, activation, level, max_levels, max_size, max_channels, margin, start_cutoff=2, critical_layers=2, window_size=6):
        super().__init__()
        self.activation = activation

        # Filter features
        self.cutoff, self.stopband, self.band_half, self.channels, self.size = self.alias_level_params(
            level, max_levels, max_size, max_channels, start_cutoff, critical_layers
        )
        self.cutoff_prev, self.stopband_prev, self.band_half_prev, self.channels_prev, self.size_prev = self.alias_level_params(
            max(level-1, 0), max_levels, max_size, max_channels, start_cutoff, critical_layers
        )

        # Filters
        self.scale_factor = 2 if self.size_prev < self.size else 1
        up_filter = self._lowpass_filter(
            window_size * self.scale_factor * 2, self.cutoff_prev, self.band_half_prev, self.size * self.scale_factor * 2
        )
        self.register_buffer("up_filter", (up_filter / up_filter.sum()) * 2 * self.scale_factor)

        down_filter = self._lowpass_filter(
            window_size * self.scale_factor, self.cutoff, self.band_half, self.size * self.scale_factor * 2
        )
        self.register_buffer("down_filter", down_filter / down_filter.sum())

        p = self.up_filter.shape[0] - (2*self.scale_factor)
        self.up_pad = ((p + 1) // 2 + (2*self.scale_factor) - 1, p // 2)

        p = self.down_filter.shape[0] - 2
        self.down_pad = ((p + 1) // 2, p // 2)
        self.margin = margin

    @staticmethod
    def alias_level_params(level, max_levels, max_size, max_channels, start_cutoff=2, critical_layers=2, base_channels=2**14):
        end_cutoff = max_size//2
        cutoff = start_cutoff * (end_cutoff / start_cutoff) ** min(level / (max_levels - critical_layers), 1)

        start_stopband = start_cutoff ** 2.1
        end_stopband = end_cutoff * (2 ** 0.3)
        stopband = start_stopband * (end_stopband/start_stopband) ** min(level / (max_levels - critical_layers), 1)

        size = 2 ** math.ceil(math.log(min(2 * stopband, max_size), 2))
        band_half = max(stopband, size / 2) - cutoff
        channels = min(round(base_channels / size), max_channels)

        return cutoff, stopband, band_half, channels, size

    def _lowpass_filter(self, n_taps, cutoff, band_half, sr):
        window = self._kaiser_window(n_taps, band_half, sr)
        ind = torch.arange(n_taps) - (n_taps - 1) / 2
        lowpass = 2 * cutoff / sr * torch.sinc(2 * cutoff / sr * ind) * window

        return lowpass

    def _kaiser_window(self, n_taps, f_h, sr):
        beta = self._kaiser_beta(n_taps, f_h, sr)
        ind = torch.arange(n_taps) - (n_taps - 1) / 2
        return torch.i0(beta * torch.sqrt(1 - ((2 * ind) / (n_taps - 1)) ** 2)) / torch.i0(torch.tensor(beta))

    def _kaiser_attenuation(self, n_taps, f_h, sr):
        df = (2 * f_h) / (sr / 2)
        return 2.285 * (n_taps - 1) * math.pi * df + 7.95


    def _kaiser_beta(self, n_taps, f_h, sr):
        atten = self._kaiser_attenuation(n_taps, f_h, sr)
        if atten > 50:
            return 0.1102 * (atten - 8.7)

        elif 50 >= atten >= 21:
            return 0.5842 * (atten - 21) ** 0.4 + 0.07886 * (atten - 21)
        else:
            return 0.0

    def forward(self, x):
        x = self._upsample(x, self.up_filter, 2*self.scale_factor, pad=self.up_pad)
        x = self.activation(x)
        x = self._downsample(x, self.down_filter, 2, pad=self.down_pad)
        if self.scale_factor > 1 and self.margin > 0:
            m = self.scale_factor * self.margin // 2
            x = x[:, :, m:-m, m:-m]
        return x

    def _upsample(self, x, kernel, factor, pad=(0, 0)):
        x = upfirdn2d(x, kernel.unsqueeze(0), up=(factor, 1), pad=(*pad, 0, 0))
        x = upfirdn2d(x, kernel.unsqueeze(1), up=(1, factor), pad=(0, 0, *pad))
        return x

    def _downsample(self, x, kernel, factor, pad=(0, 0)):
        x = upfirdn2d(x, kernel.unsqueeze(0), down=(factor, 1), pad=(*pad, 0, 0))
        x = upfirdn2d(x, kernel.unsqueeze(1), down=(1, factor), pad=(0, 0, *pad))
        return x

    def extra_repr(self):
        info_string = f'cutoff={self.cutoff}, stopband={self.stopband}, band_half={self.band_half}, channels={self.channels}, size={self.size}'
        return info_string 