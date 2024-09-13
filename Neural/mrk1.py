#!/home/akugyo/Programs/Python/PyTorch/bin/python

# Imports
import torch
from torch import nn 
from torch import optim 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm


# Neural Network
class NN(nn.Module):
	def __init__(self, input_size, num_classes):
		super(NN, self).__init__()
		self.fc1 = nn.Linear(input_size, 50)
		self.relu = nn.ReLU()  # Add this line
		self.fc2 = nn.Linear(50, num_classes)

	def forward(self, x):
		return self.fc2(self.relu(self.fc1(x)))


# model = NN(784, 10)
# x = torch.randn(64, 784)
# print(model(x).shape)

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load data
pbar = tqdm(total=4, desc="Downloading MNIST")
train_dataset = datasets.MNIST(root="data/", train=True, transform=transforms.ToTensor(), download=True)
pbar.update(1)
test_dataset = datasets.MNIST(root="data/", train=False, transform=transforms.ToTensor(), download=True)
pbar.update(1)
pbar.close()
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in tqdm(range(num_epochs), desc="Epochs"):
	for batch_idx, (data, targets) in enumerate(tqdm(train_dataloader, desc="Batches", leave=False)):
		data = data.to(device)
		targets = targets.to(device)

		data = data.reshape(data.shape[0], -1)

		scores = model(data)
		loss = criterion(scores, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

# Check Accuracy on training and test to see how good the model is
def check_accuracy(loader, model):
	num_correct = 0
	num_samples = 0

	model.eval()
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device)
			y = y.to(device)
			x = x.reshape(x.shape[0], -1)

			scores = model(x)
			_, predictions = scores.max(1)
			num_correct += (predictions == y).sum()
			num_samples += predictions.size(0)

		print(f"Acc: {num_correct}/{num_samples}={num_correct/num_samples:.2f}")


check_accuracy(train_dataloader, model)
check_accuracy(test_dataloader, model)