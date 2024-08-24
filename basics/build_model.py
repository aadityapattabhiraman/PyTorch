import os 
import torch
from torch import nn
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms


device = (
	"cuda" 
	if torch.cuda.is_available()
	else "mps"
	if torch.backends.mps.is_available()
	else "cpu"
)
print(f"Using device: {device}")


class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28 * 28, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10),
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits


model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_prob = nn.Softmax(dim=1)(logits)
y_pred = pred_prob.argmax(1)
print(f"Predicted class: {y_pred}")



# We initialize the nn.Flatten layer to convert each 2D 28x28 image into a 
# contiguous array of 784 pixel values ( the minibatch dimension (at dim=0) is 
# maintained).

# The linear layer is a module that applies a linear transformation on the input
# using its stored weights and biases

# Non-linear activations are what create the complex mappings between the model’s
# inputs and outputs. They are applied after linear transformations to introduce 
# nonlinearity, helping neural networks learn a wide variety of phenomena.

# nn.Sequential is an ordered container of modules. The data is passed through all
# the modules in the same order as defined. You can use sequential containers to 
# put together a quick network like seq_modules.

# The last linear layer of the neural network returns logits - raw values in 
# [-infty, infty] - which are passed to the nn.Softmax module. The logits are 
# scaled to values [0, 1] representing the model’s predicted probabilities for 
# each class. dim parameter indicates the dimension along which the values must 
# sum to 1.

# Many layers inside a neural network are parameterized, i.e. have associated 
# weights and biases that are optimized during training. Subclassing nn.Module 
# automatically tracks all fields defined inside your model object, and makes all
# parameters accessible using your model’s parameters() or named_parameters() 
# methods.
