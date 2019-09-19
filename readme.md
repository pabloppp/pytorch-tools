# Pytorch Tools

## Install

Requirements:
```
PyTorch >= 1.0.0
Torchivision
Numpy >= 1.0.0
```

```
# In order to install the latest (beta) use
pip install git+https://github.com/pabloppp/pytorch-tools -U

# if you want to install a specific version to avoid breaking changes (for example, v0.1.3), use 
pip install git+https://github.com/pabloppp/pytorch-tools@0.1.3 -U
```

# Current available tools

## Optimizers

Comparison table taken from https://github.com/mgrankin/over9000
And the article explaining this recent improvements https://medium.com/@lessw/how-we-beat-the-fastai-leaderboard-score-by-19-77-a-cbb2338fab5c

Dataset                               | LR Schedule| Imagenette size 128, 5 epoch | Imagewoof size 128, 5 epoch
---                                   | -- | ---                          | ---
Adam - baseline                |OneCycle| 0.8493                       | 0.6125
RangerLars (RAdam + LARS + Lookahead) |Flat and anneal| 0.8732                       | 0.6523
Ralamb (RAdam + LARS)                 |Flat and anneal| 0.8675                       | 0.6367
Ranger (RAdam + Lookahead)            |Flat and anneal| 0.8594                       | 0.5946
Novograd                              |Flat and anneal| 0.8711                       | 0.6126
Radam                                 |Flat and anneal| 0.8444                       | 0.537
Lookahead                             |OneCycle| 0.8578                       | 0.6106
Lamb                                  |OneCycle| 0.8400                       | 0.5597

### Ranger
Taken as is from https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer  
Blog post: https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d

Example of use:
```python
from torchtools.optim import Ranger

optimizer = Ranger(model.parameters())
```

### RAdam
Taken as is from https://github.com/LiyuanLucasLiu/RAdam  
Blog post: https://medium.com/@lessw/new-state-of-the-art-ai-optimizer-rectified-adam-radam-5d854730807b  
Original Paper: https://arxiv.org/abs/1908.03265  

Example of use:
```python
from torchtools.optim import RAdam, PlainRAdam, AdamW

optimizer = RAdam(model.parameters()) 
# optimizer = PlainRAdam(model.parameters()) 
# optimizer = AdamW(model.parameters()) 
```

### RangerLars (Over9000) 
Taken as is from https://github.com/mgrankin/over9000

Example of use:
```python
from torchtools.optim import RangerLars # Over9000

optimizer = RangerLars(model.parameters())
```

### Novograd 
Taken as is from https://github.com/mgrankin/over9000

Example of use:
```python
from torchtools.optim import Novograd

optimizer = Novograd(model.parameters())
```

### Ralamb 
Taken as is from https://github.com/mgrankin/over9000

Example of use:
```python
from torchtools.optim import Ralamb

optimizer = Ralamb(model.parameters())
```
 
### Lookahead
Taken as is from https://github.com/lonePatient/lookahead_pytorch  
Original Paper: https://arxiv.org/abs/1907.08610  

This lookahead can be used with any optimizer

Example of use:
```python
from torch import optim
from torchtools.optim import Lookahead

optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = Lookahead(base_optimizer=optimizer, k=5, alpha=0.5)

# for a base Lookahead + Adam you can just do:
# 
# from torchtools.optim import LookaheadAdam
```

### Lookahead
Taken as is from https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
Original Paper: https://arxiv.org/abs/1904.00962

Example of use:
```python
from torchtools.optim import Lamb

optimizer = optim.Lamb(model.parameters())
```

## LR Schedulers

### Delayed LR
Allows for a customizable number of initial steps where the learning rate remains fixed.  
After those steps the learning rate will be updated according to the supplied scheduler's policy

Example of use:
```python
from torch import optim, nn
from torchtools.lr_scheduler import DelayerScheduler

optimizer = optim.Adam(model.parameters(), lr=0.001) # define here your optimizer, the lr that you set will be the one used for the initial delay steps

delay_epochs = 10
total_epochs = 20
base_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, delay_epochs) # delay the scheduler for 10 steps
delayed_scheduler = DelayerScheduler(optimizer, total_epochs - delay_epochs, base_scheduler)

for epoch in range(total_epochs):
	# train(...)
	delayed_scheduler.step()

	# The lr will be 0.001 for the first 10 epochs, then will use the policy fro the base_scheduler for the rest of the epochs


# for a base DelayerScheduler + CosineAnnealingLR you can just do:
#
# from torchtools.lr_scheduler import DelayedCosineAnnealingLR
# scheduler = DelayedCosineAnnealingLR(optimizer, delay_epochs, cosine_annealing_epochs) # the sum of both must be the total number of epochs
```

## Activations

### Mish
Original implementation: https://github.com/digantamisra98/Mish  
Original Paper: https://arxiv.org/abs/1908.08681v1  
Implementation taken as is from https://github.com/lessw2020/mish  

Example of use:
```python
from torchtools.nn import Mish

# Then you can just use Mish as a replacement for any activation function, like ReLU
```

## Layers

### SimpleSelfAttention
Implementation taken as is from https://github.com/sdoria/SimpleSelfAttention

Example of use:
```python
from torchtools.nn import SimpleSelfAttention

# The input of the layer has to at least have 3 dimensions (B, C, N), 
# the attention will be performed in the 2nd dimension.
# 
# For images, the input will be internally reshaped to 3 dimensions,
# and reshaped back to the original shape before returning it
```

### PixelNorm
Inspired from https://github.com/github-pengge/PyTorch-progressive_growing_of_gans

Example of use:
```python
from torchtools.nn import PixelNorm

model = nn.Linear(
	nn.Conv2d(...),
	PixelNorm(),
	nn.ReLU()
)

# It doesn't require any parameter, it just performs a simple element-wise normalization
# x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
#
# Just use it as a regular layer, generally after convolutions and before ReLU
# (warning) since it performs a srtq root it's pretty slow if the layer sizes are big
```

## Criterions

### Gradient Penalty (for WGAN-GP)
Implementation taken with minor changes from https://github.com/caogang/wgan-gp  
Original paper https://arxiv.org/pdf/1704.00028.pdf

Example of use:
```python
from torchtools.nn import GPLoss
# This criterion defines the gradient penalty for WGAN GP
# For an example of a training cycle refer to this repo https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L185

discriminator = ...
gpcriterion = GPLoss(discriminator) # l = 10 by default

gradient_penalty = gpcriterion(real_data, fake_data)
discriminator_loss = ... + gradient_penalty # add the gp component to the Wasserstein loss
```

### Total Variation Loss
Total Variation denoising https://www.wikiwand.com/en/Total_variation_denoising  

Example of use:
```python
# This loss (or regularization) is usefull for removing artifacts and noise in generated images.  
# It's widely used in style transfer.
from torchtools.nn import TVLoss

tvcriterion = TVLoss(discriminator) # reduction = 'sum' and alpha = 1e-4 by default

G = ... # output image
tv_loss = tvcriterion(G)
loss = ... + tv_loss # add the tv loss component to your reconstruction loss
```


## Vector Quantization
### VectorQuantize: Encodding based quantization [(source)](torchtools/vq.py#L5)
This transforms any tensor to its quantized version using a codebook of embeddings.  
It uses a traight-forward approach for applying the gradients.  
Passing a tensor trough the **VectorQuantize** module will return a new tensor with the same dimension but changing each one of the tensors of the last dimension by the nearest neighbor from the codebook, which has a limited number of values, thus quantizing the tensor.

For the quantization it relies in a differentiable function that you can see [here](torchtools/functional/vq.py#L4)

The output of the model is a quantized tensor, as well as a Touple of the loss components of the codebook (needed for training), and the indices of the quantized vectors in the form: `qx, (vq_loss, commit_loss), indices`

When **creating a new instance of the module**, it accepts the following parameters: 
  - **embedding_size**: the size of the embeddings used in the codebook, should match the last dimension of the tensor you want to quantize
  - **k**: the size of the codebook, or number of embeddings. 
  - **ema_decay** (default=0.99): the Exponentially Moving Average decay used (this only will be used if ema_loss is True)
  - **ema_loss** (default=False): Enables Exponentially Moving Average update of the codebook (instead of relying on gradient descent as EMA converges faster) 
  
When **calling the forward method** of the module, it accepts the following parameters:
  - **x**: this is the tensor you want to quantize, make sure the dimension that you want to quantize (by default is the last one) matches embedding_size defined when instantiating the module
  - **get_losses** (default=True): when False, the vq_loss and commit_loss components of the output will both be None, this should speed up a little bit the model when used for inference.
  - **dim** (default=-1): The dimension across which the input should be quantized.

Example of use:
```python
from torchtools.nn import VectorQuantize

e = torch.randn(1, 16, 16, 8) # create a random tensor with 8 as its last dimension size
vquantizer = VectorQuantize(8, k=32, ema_loss=True) # we create the module with embedding size of 8, a codebook of size 32 and make the codebook update using EMA
qe, (vq_loss, commit_loss), indices = vquantizer.forward(e) # we quantize our tensor while also getting the loss components and the indices

# NOTE While the model is in training mode, the codebook will always be updated when calling the forward method, in order to freeze the codebook for inference put it in evaluation mode with 'vquantizer.eval()'

# NOTE 2 In order to update the module properly, add the loss components to the final model loss before calling backward(), if you set ema_loss to true you only need to add the commit_loss to the total loss, an it's usually multiplied by a value between 0.1 and 2, being 0.25 a good default value

loss = ... # whatever loss you have for your final output
loss += commit_loss * 0.25
# loss += vq_loss # only if you didn't set the ema_loss to True

...
loss.backward()
optimizer.step()

```

--- 

### Binarize: binarize the input tensor [(source)](torchtools/vq.py#L55)
This transfors the values of a tensor into 0 and 1 depending if they're above or below a specified threshold.
It uses a traight-forward approach for applying the gradients, so it's effectively differentiable.

For the quantization it relies in a differentiable function that you can see [here](torchtools/functional/vq.py#L36)

Example of use:
```python
from torchtools.nn import Binarize

e = torch.randn(8, 16) # create a random tensor with any dimension

binarizer = Binarize(threshold=0.5) # you can set the threshold you want, for example if your output was passed through a tanh activation, 0 might be a better theshold since tanh outputs values between -1 and 1

bq = binarizer(e) # will return a tensor with the same shape as e, but full of 0s and 1s
```
