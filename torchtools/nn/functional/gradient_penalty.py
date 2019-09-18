####
# CODE TAKEN WITH FEW MODIFICATIONS FROM https://github.com/caogang/wgan-gp
# ORIGINAL PAPER https://arxiv.org/pdf/1704.00028.pdf
####

import torch
from torch import autograd

def gradient_penalty(netD, real_data, fake_data, l=10):
    batch_size = real_data.size(0)
    alpha = real_data.new_empty((batch_size, 1, 1, 1)).uniform_(0, 1)
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=real_data.new_ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * l

    return gradient_penalty