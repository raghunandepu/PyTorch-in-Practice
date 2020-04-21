import torch

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


# Addition of two tensors
x = torch.rand(2,2)
y = torch.rand(2,2)
z = torch.add(x,y) # same as x + y
#print(z)

# Inplace addition
# Any function with trailing underscore(_) in pytorch does inplace operation
print(y)
y.add_(x)
print(y)