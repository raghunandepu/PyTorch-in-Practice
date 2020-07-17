# Package to calcuate gradients in pytorch

import torch
x = torch.randn(3, requires_grad=True) # By default grad is false
print(x)

# Whenever we do operations with tensors, pytorch will create a computational graph for us.

y = x + 2
print(y) # observe the grad_fn. It is add because of addition.
z = y*y*2
print(z) # observe the grad_fn. It is add because of addition.
