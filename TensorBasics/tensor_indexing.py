import torch

# ===========================================
#               Tensor Indexing
# ===========================================

batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0])
print(x[0].shape) # x[0,:]