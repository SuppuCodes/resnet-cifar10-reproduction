from models.resnet import ResNet18
import torch

# create model
model = ResNet18()

# create dummy input (batch_size, channels, height, width)
x = torch.randn(1, 3, 32, 32)

# forward pass
y = model(x)

print("Output shape:", y.shape)