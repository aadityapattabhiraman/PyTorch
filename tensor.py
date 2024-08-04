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

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

some_tensor = torch.rand(3, 4)
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}")

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)
print(torch.mm(tensor_A, tensor_B.T))

tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16)

x = torch.arange(1., 8.)
x_reshaped = x.reshape(1, 7)
print(x_reshaped, x_reshaped.shape)

x_stacked = torch.stack([x, x, x, x], dim=0) 
print(x_stacked)

x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")

x_original = torch.rand(size=(224, 224, 3))
x_permuted = x_original.permute(2, 0, 1)
print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")