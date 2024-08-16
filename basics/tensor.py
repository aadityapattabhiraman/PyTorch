import torch  


## Print Torch Version
print(torch.__version__)

## Creating a tensor
# Scalar to be precise
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)

# Vector
vector = torch.tensor([7, 7])
print(vector)
print(vector.shape)

# Matrix
matrix = torch.tensor([[7, 8],
                       [9, 10]])
print(matrix)
print(matrix.ndim)
print(matrix.shape)

# Tensor
tensor = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(tensor)
print(tensor.ndim)
print(tensor.shape)

# Random Tensor
random_tensor = torch.rand(size=(3, 4))
print(random_tensor, random_tensor.dtype)

random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

# Zero tensor
zeros = torch.zeros(size=(3, 4))
print(zeros, zeros.dtype)

# One Tensor
ones = torch.ones(size=(3, 4))
print(ones, ones.dtype)

# Range tensor (definitely not called that though)
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)

ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeros)

# DataType of Tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, 
                               device=None,
                               requires_grad=False)

print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

some_tensor = torch.rand(3, 4)
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}")

# Matrix Multiplication
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)
print(torch.mm(tensor_A, tensor_B.T))

# Changing datatype
tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16)

# Reshaping a tensor
x = torch.arange(1., 8.)
x_reshaped = x.reshape(1, 7)
print(x_reshaped, x_reshaped.shape)

# Stack a tensor
x_stacked = torch.stack([x, x, x, x], dim=0) 
print(x_stacked)

# Squeeze a tensor
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")

# Permute a tensor
x_original = torch.rand(size=(224, 224, 3))
x_permuted = x_original.permute(2, 0, 1)
print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")