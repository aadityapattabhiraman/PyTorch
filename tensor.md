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