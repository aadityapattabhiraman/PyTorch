import torch
import os 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image



# Loading the dataset
training_data = datasets.FashionMNIST(
	root="data", train=True, download=True, transform=ToTensor()
)

test_data = datasets.FashionMNIST(
	root="data", train=False, download=True, transform=ToTensor()
)


# Acustom dataset class must implement the following three functions
class CustomImageDataset(Dataset):
	def __init__(self, annotations_file, img_dir, transform=None,
				 target_transform=None):
		self.img_labels = pd.read_csv(annotations_file)
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	# Finds the len
	def __len__(self):
		return len(self.img_labels)

	# Returns a sample from the dataset given the index
	def __get_item__(self, idx):
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
		image = read_image(img_path)
		label = self.img_labels.iloc[idx, 1]

		if self.transform:
			image = self.transform(image)

		if self.target_transform:
			label = self.target_transform(label)

		return image, label


# Preparing the dataloaders
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through the dataloader
train_features, train_labels = next(iter(train_dataloader))
img = train_features[0].squeeze()
label = train_labels[0]