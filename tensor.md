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