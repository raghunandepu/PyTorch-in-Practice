import torch

# ===========================================
#               Tensor Indexing
# ===========================================

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0])
print(x[0].shape) # x[0,:]

print(x[2, 0:10])

x[0, 0] = 100
print(x)
indices = [2, 5, 6]
print(x[indices])

x = torch.rand((3, 5))
print(x.shape)

rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols].shape)


# More Advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0])

# Useful operations
print(torch.where(x > 5, x, x*2))
print(torch.tensor([0, 0, 0, 2, 2, 3, 4]).unique())
print(x.ndimension())
print(x.numel()) # count the number of elements in x


