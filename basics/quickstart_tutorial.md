### Quickstart Tutorial  
import statements  
```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```
Downloading training and testing data  
```python
training_data = datasets.FashionMNIST(
	root="data", train=True, download=True, transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
	root="data", train=False, download=True, transform=ToTensor(),
)
```
Creating dataloaders  
```python
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size)
test_dataloader = DataLoader(test_data, batch_size)
```
Printing shapes
```python
for X, y in test_dataloader:
	print(f"Shape of X [N, C, H, W]: {X.shape}")
	print(f"Shape of y: {y.shape} {y.dtype}")
	break
```
**Output:**
```python
Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
Shape of y: torch.Size([64]) torch.int64
```