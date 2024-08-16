### Neural Network Creation

#### Import Statements
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
```
- `nn` handles neural network construction  
- `optim` focuses on optimization algorithms  
- `functional` provides various loss functions  
- `datasets` facilitates working with PyTorch datasets  
- `transforms` applies different transformations  
- `dataloader` simplifies data loading


Training loop song
```
For an epoch in a range
Call model dot train
Do the forward pass
Calculate the loss
Optimizer zero grad
Lossssss backward
Optimizer step step step

Test time!
Call model dot eval
With torch inference mode
Do the forward pass
Calculate the loss
```