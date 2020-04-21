import torch
import numpy as np

# To create a 3D tensor
x = torch.empty(3,2,3)
#print(x)

# Create a tensor with random values
x = torch.rand(3,2)

# Tensor of zeros
x = torch.zeros(2,2)
#print(x)

# Tensor of ones
x = torch.ones(2,2, dtype=torch.int) # dtype is float by default
#print(x.dtype)
#print(x.size())

# To create a tensor with list of values
x = torch.tensor([2.5, 3.2])
#print(x)


# Addition of two tensors (Element wise)
x = torch.rand(2,2)
y = torch.rand(2,2)
z = torch.add(x,y) # same as x + y
#print(z)

# Inplace addition(Element wise)
# Any function with trailing underscore(_) in pytorch does inplace operation
#print(y)
#y.add_(x)
#print(y)

#subtraction
#print(z)
#z = torch.sub(x,y)
#print(z)

#inplace subtraction
#print(y)
#y.sub_(x)
#print(y)

#Slicing
"""x = torch.rand(5,3)
print(x)"""
#print(x[1,:])

#specific item in tensor
#print(x[1,1].item())

# Reshaping tensor
#y = x.view(15) # Convering 2D(5,3) vector of x as 1D in y
#print(y)

#if we put -1 as dimension for first number, torch automatically detects the correct number
"""x = torch.rand(4,4)
print(x)
y = x.view(-1,8)
print(y)
print(y.size())"""

# Converting torch tensor to numpy array to and viceversa
"""a = torch.ones(5)
print(a)
print(type(a))
b = a.numpy()
print(b)
print(type(b))
#Both share same memory location in RAM. Be careful.
a.add_(1)
print(a)
print(b)"""


#Convert the numpy array to torch tensor
"""a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)"""
#Both share same memory location in RAM. Be careful.

if torch.cuda.is_available(): # False in MAC
  device = torch.device("cuda")
  x = torch.ones(5, device = device)
  y = torch.ones(5)
  y = y.to(device)
  z = x + y
  z = z.to("cpu")
  
# This will tell pytorch to calcuate gradients later during optimizations.
x = torch.ones(5, requires_grad=True)
print(x)