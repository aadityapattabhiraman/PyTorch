import torch  


## Print Torch Version
print(torch.__version__)

## Creating a tensor
# Scalar to be precise
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)

vector = torch.tensor([7, 7])
print(vector)
print(vector.shape)

matrix = torch.tensor([[7, 8],
                       [9, 10]])
print(matrix)
print(matrix.ndim)
print(matrix.shape)