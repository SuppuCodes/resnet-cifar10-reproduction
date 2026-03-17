import torch
import torch.nn as nn
import torch.optim as optim

from models.resnet import ResNet18
from data.dataset_loader import load_cifar10


# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Starting training script...")
# load dataset
train_loader, test_loader = load_cifar10()
print("Dataset loaded")

# initialize model
model = ResNet18().to(device)
print("Model initialized")

# loss function
criterion = nn.CrossEntropyLoss()


# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


# number of epochs
num_epochs = 20

for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} Loss {loss.item():.4f}")
 # end of epoch print
print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")