## Tensor Notes


### Import Statement
Statement to be used to import PyTorch
```python
import torch
```  

### PyTorch version
Statement to be used to print the PyTorch version  
```python
print(torch.__version__)
```  
**Output:**  
```python
2.4.0+cu124
```

### Tensor Creation  
#### 1. Scalar  

Creating a scalar  
A scalar is a single number and in tensor language it's a zero dimension tensor.  
```python
scalar = torch.tensor(7)
print(scalar)
```
**Output:**
```python
tensor(7)
```  

Checking the dimension  
```python
print(scalar.ndim)
```  
**Output:**
```python
0
```  

#### 2. Vector  

Creating a Vector  
A vector is a single dimension tensor but can contain many numbers.  
```python
vector = torch.tensor([7, 7])
print(vector)
```
**Output:**
```python
tensor([7, 7])
```

Checking the dimension  
```python
print(vector.ndim)
```  
**Output:**
```python
1
```  
Checking the shape  
```python
print(vector.shape)
```  
**Output:**  
```python
torch.Size([2])
```  

#### 3. Matrix  

Creating a matrix  
Matrices are as flexible as vectors, except they've got an extra dimension.  
```python
matrix = torch.tensor([[7, 8],
                       [9, 10]])
print(matrix)
```  
**Output:**
```python
tensor([[ 7,  8],
        [ 9, 10]])
```

Checking the dimension  
```python
print(matrix.ndim)
```  
**Output:**
```python
2
```
Checking the shape
```python
print(matrix.shape)
```
**Output:**  
```python
torch.Size([2, 2])
```

#### 4. Tensor

Creating a tensor
Tensors are an n-dimensional array of numbers
```python
tensor = torch.tensor([[[1, 2, 3],
                        [3, 6, 9],
                        [2, 4, 5]]])
print(tensor)
```
**Output:**
```python
tensor([[[1, 2, 3],
         [3, 6, 9],
         [2, 4, 5]]])
```

Checking the dimension
```python
print(tensor.ndim)
```
**Output:**
```python
3
```

Checking the shape
```python
print(tensor.shape)
```
**Output:**
```python
torch.Size([1, 3, 3])
```

![How shapes are calculated!](Picture/00-pytorch-different-tensor-dimensions.png "How shapes are claculated")

Let's summarise.

| Name | What is it? | Number of dimensions | Lower or upper (usually/example) |
| ----- | ----- | ----- | ----- |
| **scalar** | a single number | 0 | Lower (`a`) | 
| **vector** | a number with direction (e.g. wind speed with direction) but can also have many other numbers | 1 | Lower (`y`) |
| **matrix** | a 2-dimensional array of numbers | 2 | Upper (`Q`) |
| **tensor** | an n-dimensional array of numbers | can be any number, a 0-dimension tensor is a scalar, a 1-dimension tensor is a vector | Upper (`X`) | 

I may not follow the uppercase thing cause i dont like using uppercase.  

### Random Tensors
Creating a random tensor

```python
random_tensor = torch.rand(size=(3, 4))
random_tensor, random_tensor.dtype
```
**Output:**
```python
tensor([[0.3863, 0.8313, 0.8441, 0.8897],
        [0.6326, 0.4750, 0.2865, 0.2915],
        [0.2821, 0.5702, 0.9430, 0.2756]]) torch.float32
```

Creating a random tensor of size (224, 224, 3) A Common image size
```python
random_image_size_tensor = torch.rand(size=(224, 224, 3))
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)
```
**Output:**
```python
torch.Size([224, 224, 3]) 3
```

### Zeros and Ones

Creating a zero tensor
```python
zeros = torch.zeros(size=(3, 4))
print(zeros, zeros.dtype)
```
**Output:**
```python
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]]) torch.float32
```

Creating a ones tensor
```python
ones = torch.ones(size=(3, 4))
print(ones, ones.dtype)
```
**Output:**
```python
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]]) torch.float32
```

### Range of Tensors

Sometimes you might want a range of numbers, such as 1 to 10 or 0 to 100.
```python
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten)
```
**Output:**
```python
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

Create a tensor like another tensor
```python
ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeros)
```
**Output:**
```python
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```

### Tensor DataTypes
Check this for more info [PyTorch] (https://pytorch.org/docs/stable/tensors.html#data-types)

```python
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,
                               device=None, 
                               requires_grad=False)

print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)
```
**Output:**
```python
torch.Size([3]) torch.float32 cpu
```

### Getting info from tensors  
* `shape` - what shape is the tensor? (some operations require specific shape rules)
* `dtype` - what datatype are the elements within the tensor stored in?
* `device` - what device is the tensor stored on? (usually GPU or CPU)

```python
some_tensor = torch.rand(3, 4)

# Find out details about it
print(some_tensor)
print(f"Shape of tensor: {some_tensor.shape}")
print(f"Datatype of tensor: {some_tensor.dtype}")
print(f"Device tensor is stored on: {some_tensor.device}")
```
**Output:**
```python
tensor([[0.0768, 0.1223, 0.2892, 0.7063],
        [0.4531, 0.8558, 0.6990, 0.5572],
        [0.4770, 0.5193, 0.3438, 0.5791]])
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

### Manipulating tensors (tensor operations)
* Addition
* Substraction
* Multiplication (element-wise)
* Division
* Matrix multiplication

### Matrix multiplication is ALL YOU NEED
![Matrix multiplication!](Picture/00_matrix_multiplication_is_all_you_need.jpeg" How shapes are claculated")

```python
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)
print(torch.mm(tensor_A, tensor_B.T))
```
**Output:**
```python
tensor([[ 27.,  30.,  33.],
        [ 61.,  68.,  75.],
        [ 95., 106., 117.]])
```
Changing the datatype
```python
tensor = torch.arange(10., 100., 10.)
print(tensor.dtype)
tensor_float16 = tensor.type(torch.float16)
print(tensor_float16)
```
**Output:**
```python
torch.float32
tensor([10., 20., 30., 40., 50., 60., 70., 80., 90.], dtype=torch.float16)
```

### Reshape

Reshaping a tensor
```python
x = torch.arange(1., 8.)
print(x, x.shape)
x_reshaped = x.reshape(1, 7)
print(x_reshaped, x_reshaped.shape)
```
**Output:**
```python
tensor([1., 2., 3., 4., 5., 6., 7.]) torch.Size([7])
tensor([[1., 2., 3., 4., 5., 6., 7.]]) torch.Size([1, 7])
```

### Stack

Stack a tensor on top of each other
```python
x_stacked = torch.stack([x, x, x, x], dim=0) 
print(x_stacked)
```
**Output:**
```python
tensor([[1., 2., 3., 4., 5., 6., 7.],
        [1., 2., 3., 4., 5., 6., 7.],
        [1., 2., 3., 4., 5., 6., 7.],
        [1., 2., 3., 4., 5., 6., 7.]])
```

### Squeeze

Remove single dimension from a tensor
```python
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
```
**Output:**
```python
New tensor: tensor([1., 2., 3., 4., 5., 6., 7.])
```
Unsqueeze does the opposite
Permute does the permutation
```python
x_original = torch.rand(size=(224, 224, 3))
x_permuted = x_original.permute(2, 0, 1)
print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")
```
**Output:**
```python
Previous shape: torch.Size([224, 224, 3])
New shape: torch.Size([3, 224, 224])
```

## There are a lot of other features in PyTorch, but font care cause for one time use you can gpt it.

### We are done.