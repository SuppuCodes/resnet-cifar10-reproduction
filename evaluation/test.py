import torch
from models.resnet import ResNet18
from data.dataset_loader import load_cifar10

from sklearn.metrics import confusion_matrix
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Loading dataset...")
train_loader, test_loader = load_cifar10()


print("Loading model...")
model = ResNet18().to(device)

# load trained weights
model.load_state_dict(torch.load("resnet18_cifar10.pth"))

model.eval()


correct = 0
total = 0

all_preds = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")


# confusion matrix
cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix:")
print(cm)