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

tensor = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(tensor)
print(tensor.ndim)
print(tensor.shape)

random_tensor = torch.rand(size=(3, 4))
print(random_tensor, random_tensor.dtype)

random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

zeros = torch.zeros(size=(3, 4))
print(zeros, zeros.dtype)

ones = torch.ones(size=(3, 4))
print(ones, ones.dtype)

zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)

ten_zeros = torch.zeros_like(input=zero_to_ten) # will have same shape
print(ten_zeros)