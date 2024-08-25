import torch
import torchvision.models as models


model = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model.state_dict(), "model.pth")

model = models.vgg16()
model.load_state_dict(torch.load('model.pth', weights_only=True))
torch.save(model, 'model.pth')
model = torch.load('model.pth')