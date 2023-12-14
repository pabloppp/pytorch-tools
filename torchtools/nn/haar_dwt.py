import torch
from torch import nn

# Taken almost as is from https://github.com/bes-dev/haar_pytorch
class HaarForward(nn.Module):
    """
    Performs a 2d DWT Forward decomposition of an image using Haar Wavelets
    set beta=1 for regular haard dwt, with beta=2 we make a magnitude preserving dwt
    """
    def __init__(self, beta=2):
        super().__init__()
        self.alpha = 0.5
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a 2d DWT Forward decomposition of an image using Haar Wavelets

        Arguments:
            x (torch.Tensor): input tensor of shape [b, c, h, w]

        Returns:
            out (torch.Tensor): output tensor of shape [b, c * 4, h / 2, w / 2]
        """

        ll = self.alpha/self.beta * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] + x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
        lh = self.alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] - x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
        hl = self.alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] + x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
        hh = self.alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] - x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
        return torch.cat([ll,lh,hl,hh], axis=1)


class HaarInverse(nn.Module):
    """
    Performs a 2d DWT Inverse reconstruction of an image using Haar Wavelets
    set beta=1 for regular haard dwt, with beta=2 we make a magnitude preserving dwt
    """
    def __init__(self, beta=2):
        super().__init__()
        self.alpha = 0.5
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a 2d DWT Inverse reconstruction of an image using Haar Wavelets

        Arguments:
            x (torch.Tensor): input tensor of shape [b, c, h, w]

        Returns:
            out (torch.Tensor): output tensor of shape [b, c / 4, h * 2, w * 2]
        """
        assert x.size(1) % 4 == 0, "The number of channels must be divisible by 4."
        size = [x.shape[0], x.shape[1] // 4, x.shape[2] * 2, x.shape[3] * 2]
        f = lambda i: x[:, size[1] * i : size[1] * (i + 1)]
        out = torch.zeros(size, dtype=x.dtype, device=x.device)
        out[:,:,0::2,0::2] = self.alpha * (f(0)*self.beta + f(1) + f(2) + f(3))
        out[:,:,0::2,1::2] = self.alpha * (f(0)*self.beta + f(1) - f(2) - f(3))
        out[:,:,1::2,0::2] = self.alpha * (f(0)*self.beta - f(1) + f(2) - f(3))
        out[:,:,1::2,1::2] = self.alpha * (f(0)*self.beta - f(1) - f(2) + f(3))
        return out