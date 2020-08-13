import torch

# ========================================
#          Tensor Reshaping
# ========================================

x = torch.arange(9)
x_3x3 = x.view(3,3)  # converting x to 3X3 matrix
print(x_3x3)

# another way to convert x to 3X3
x_3x3 = x.reshape(3, 3)   # view is better than reshape in terms of performance but using reshape is safe bet.
print(x_3x3)

y = x_3x3.t()  # transpose
print(y.contiguous().view(9))  # tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
print(torch.cat((x1, x2), dim=0).shape)  # concat --> result: [4, 5]
print(torch.cat((x1, x2), dim=1).shape)  # concat --> result: [2, 10]


z = x1.view(-1)  # flatten x1
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)  # flattening dimensions except batch (first dimension)
print(z.shape)  # [64, 10]


z = x.permute(0, 2, 1,)  # reshaping x [64, 2, 5] dimensions
print(z.shape)  # [64, 5, 2]

x = torch.arange(10)
print(x.shape)  # [10]
print(x.unsqueeze(0).shape)  # [1, 10]
print(x.unsqueeze(1).shape)  # [10, 1]

x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1X1X10
print(x.shape)

z = x.squeeze(1)
print(z.shape)
