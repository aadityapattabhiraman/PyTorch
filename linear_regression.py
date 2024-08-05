import torch 
from torch import nn  
import matplotlib.pyplot as plt


# Hyperparameters
weight = 0.7 
bias = 0.3 

# Make some data
start = 0
end = 1 
step = 0.02 
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * x + bias

# Split the data into training and testing
train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]


# Visualise the data else you wont learn ml
def plot_predictions(train_data=x_train, train_labels=y_train, 
                     test_data=x_test, test_labels=y_test, 
                     predictions=None):
	
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
    	plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()

plot_predictions()


# Model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float),
        							requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), 
        						 requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(42)
model_0 = LinearRegression()

with torch.inference_mode():
    y_preds = model_0(x_test)

plot_predictions(predictions=y_preds)
# Loss and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

epochs = 100

train_loss_values = []
test_loss_values = []
epoch_count = []

# Training loop
for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()

    with torch.inference_mode():
        test_pred = model_0(x_test)
        test_loss = loss_fn(test_pred, y_test.type(torch.float))

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss:",
            	f"{test_loss} ")

# Plot again
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# Evaluate the model
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(x_test)

plot_predictions(predictions=y_preds)
# save
torch.save(model_0, "models/trial_model.pth")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Gpu model
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, 
                                      out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Initialising
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1, model_1.state_dict()
model_1.to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

torch.manual_seed(42)
epochs = 1000 
x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# Training loop
for epoch in range(epochs):

    model_1.train()
    y_pred = model_1(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_1.eval()

    with torch.inference_mode():
        test_pred = model_1(x_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

# Inference
model_1.eval()
with torch.inference_mode():
    y_preds = model_1(x_test)

plot_predictions(predictions=y_preds.cpu())
# Save
torch.save(model_0, "models/lesgo_model.pth")