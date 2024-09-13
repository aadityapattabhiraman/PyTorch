#!/home/akugyo/Programs/Python/PyTorch/bin/python

# Imports
import torch
from torch import nn 
from torch import optim 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
from time import time


# Neural Network
class NN(nn.Module):
	def __init__(self, input_size, num_classes):
		super(NN, self).__init__()
		self.fc1 = nn.Linear(input_size, 50)
		self.relu = nn.ReLU()  
		self.fc2 = nn.Linear(50, num_classes)

	def forward(self, x):
		return self.fc2(self.relu(self.fc1(x)))


# CNN
class CNN(nn.Module):
	def __init__(self, in_channels=1, num_classes=10):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(
			in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), 
			padding=(1,1)
		)
		self.pool = nn.MaxPool2d(
			kernel_size=(2, 2), stride=(2, 2)
		)
		self.conv2 = nn.Conv2d(
			in_channels=8, out_channels=16, kernel_size=(3, 3),stride=(1, 1),
			padding=(1, 1)
		)
		self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
		self.relu = nn.ReLU()

	def forward(self, x):
		return self.fc1(self.pool(self.conv2(self.pool(self.conv1(x)))).reshape(x.shape[0], -1))


# model = CNN()
# x = torch.randn(64, 1, 28, 28)
# print(model(x).shape)
start = time()
# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load data
pbar = tqdm(total=2, desc="Downloading MNIST")
train_dataset = datasets.MNIST(
	root="data/", train=True, transform=transforms.ToTensor(), download=True
	)
pbar.update(1)
test_dataset = datasets.MNIST(
	root="data/", train=False, transform=transforms.ToTensor(), download=True
	)
pbar.update(1)
pbar.close()
train_dataloader = DataLoader(
	dataset=train_dataset, batch_size=batch_size, shuffle=True
	)
test_dataloader = DataLoader(
	dataset=test_dataset, batch_size=batch_size, shuffle=True
	)

# Initialize network
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in tqdm(range(num_epochs), desc="Epochs"):
	for batch_idx, (data, targets) in enumerate(tqdm(train_dataloader, desc="Batches", leave=False)):
		data = data.to(device)
		targets = targets.to(device)

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

			scores = model(x)
			_, predictions = scores.max(1)
			num_correct += (predictions == y).sum()
			num_samples += predictions.size(0)

		print(f"Acc: {num_correct}/{num_samples}={num_correct/num_samples:.2f}")


check_accuracy(train_dataloader, model)
check_accuracy(test_dataloader, model)
end = time()
print(end - start)