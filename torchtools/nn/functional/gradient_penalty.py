####
# CODE TAKEN WITH FEW MODIFICATIONS FROM https://github.com/caogang/wgan-gp
# ORIGINAL PAPER https://arxiv.org/pdf/1704.00028.pdf
####

import torch
from torch import autograd

def gradient_penalty(netD, real_data, fake_data, l=10):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand((batch_size, real_data.nelement()//batch_size)).contiguous().view(batch_size, 16, 128, 128)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * l
    return gradient_penalty