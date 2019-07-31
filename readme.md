# Pytorch Tools

## Install
```
# In order to install the latest (beta) use
pip install git+https://github.com/pabloppp/pytorch-tools -U

# if you want to install a specific version to avoid breaking changes (for example, v0.1.3), use 
pip install git+https://github.com/pabloppp/pytorch-tools@0.1.3 -U
```

## Current available tools
 
### Vector Quantization
#### VectorQuantize: Encodding based quantization [(source)](torchtools/vq.py#L5)
This transforms any tensor to its quantized version using a codebook of embeddings.  
It uses a traight-forward approach for applying the gradients.  
Passing a tensor trough the **VectorQuantize** module will return a new tensor with the same dimension but changing each one of the tensors of the last dimension by the nearest neighbor from the codebook, which has a limited number of values, thus quantizing the tensor.

For the quantization it relies in a differentiable function that you can see [here](torchtools/functional/vq.py#L4)

The output of the model is a quantized tensor, as well as a Touple of the loss components of the codebook (needed for training) in the form: `qx, (vq_loss, commit_loss)`

--- 

When **creating a new instance of the module**, it accepts the following parameters: 
  - **embedding_size**: the size of the embeddings used in the codebook, should match the last dimension of the tensor you want to quantize
  - **k**: the size of the codebook, or number of embeddings. 
  - **ema_decay** (default=0.99): the Exponentially Moving Average decay used (this only will be used if ema_loss is True)
  - **ema_loss** (default=False): Enables Exponentially Moving Average update of the codebook (instead of relying on gradient descent as EMA converges faster) 
  
When **calling the forward method** of the module, it accepts the following parameters:
  - **x**: this is the tensor you want to quantize, make sure the last dimension of the tensor matches embedding_size defined when instantiating the module
  - **get_losses** (default=True): when False, the vq_loss and commit_loss components of the output will both be None, this should speed up a little bit the model when used for inference.

Example of use:
```python
from torchtools.vq import VectorQuantize

e = torch.randn(1, 16, 16, 8) # create a random tensor with 8 as its last dimension size
vquantizer = VectorQuantize(8, k=32, ema_loss=True) # we create the module with embedding size of 8, a codebook of size 32 and make the codebook update using EMA
qe, (vq_loss, commit_loss) = vquantizer.forward(e) # we quantize our tensor while also getting the loss components

# NOTE While the model is in training mode, the codebook will always be updated when calling the forward method, in order to freeze the codebook for inference put it in evaluation mode with 'vquantizer.eval()'

# NOTE 2 In order to update the module properly, add the loss components to the final model loss before calling backward(), if you set ema_loss to true you only need to add the commit_loss to the total loss, an it's usually multiplied by a value between 0.1 and 2, being 0.25 a good default value

loss = ... # whatever loss you have for your final output
loss += commit_loss * 0.25
# loss += vq_loss # only if you didn't set the ema_loss to True

...
loss.backward()
optimizer.step()

```

#### Binarize: binarize the input tensor [(source)](torchtools/vq.py#L55)
This transfors the values of a tensor into 0 and 1 depending if they're above or below a specified threshold.
It uses a traight-forward approach for applying the gradients, so it's effectively differentiable.

For the quantization it relies in a differentiable function that you can see [here](torchtools/functional/vq.py#L36)

Example of use:
```python
from torchtools.vq import Binarize

e = torch.randn(8, 16) # create a random tensor with any dimension

binarizer = Binarize(threshold=0.5) # you can set the threshold you want, for example if your output was passed through a tanh activation, 0 might be a better theshold since tanh outputs values between -1 and 1

bq = binarizer(e) # will return a tensor with the same shape as e, but full of 0s and 1s
```
