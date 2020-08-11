import torch

# ==================================================================
#             Tensor Math & Comparison Operators
# ==================================================================

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition

z1 = torch.empty(3)
torch.add(x,y, out=z1)
print(z1)

z2 = torch.add(x,y)
print(z2)

# same as above
z3 = x + y
print(z3)

# Subtraction

z = x - y

# Division

z = torch.true_divide(x,y) # element wise division
print(z)


# inplace operations
t = torch.zeros(3)
t.add_(x)     # here t will get updated. Whenever we see_ as suffix, it is inplace.
print(t)

# same as above
t += x

# Exponentiation
z = x.pow(2)
z = x ** 2

# simple comparison
z = x > 0
print(z)

# Matrix multiplication
x1 = torch.rand((2,4))
x2 = torch.rand((4,3))
x3 = torch.mm(x1, x2) # 2X3
print(x3)
#same as above
x3 = x1.mm(x2)

# Matrix Exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

# Element wise multiplication
z = x * y
print(z)

# Dot Product -> Element wise multiplication and then addition
z = torch.dot(x, y)
print(z)

# Batch Matrix multiplication - 3 dimensions
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)  # Result size ->(batch, n, p)

# Example of Broadcasting
x1 = torch.rand((4, 5))
x2 = torch.rand((1, 5))  # copied as per x1 while doing operations to match

z = x1 - x2
print(z)

# Other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0)
z = torch.mean(x.float(), dim=0)
z = torch.eq(x,y)
sorted_y,indices = torch.sort(y, dim=0, descending=False)

# values less than x are set to 0 and or which are greater than 10
z = torch.clamp(x, min=0, max=10)

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x)
print(z)

