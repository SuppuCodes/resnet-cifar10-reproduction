import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from models.resnet import ResNet18
from data.dataset_loader import load_cifar10


# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load dataset
#train_loader, test_loader = load_cifar10(use_augmentation=False)
train_loader, test_loader = load_cifar10(use_augmentation=True)  


# initialize model
model = ResNet18().to(device)


# loss function
criterion = nn.CrossEntropyLoss()
#optimizer (SGD experiment)
""" optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
) """

# optimizer (Adam experiment)
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)

#Remove scheduler for Adam
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# TensorBoard writer
#writer = SummaryWriter("runs/sgd_no_aug")
#writer = SummaryWriter("runs/sgd_aug")
writer = SummaryWriter("runs/adam_aug")


# evaluation function
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# number of epochs
num_epochs = 10


# training loop
for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # TensorBoard: batch loss
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("Loss/train_batch", loss.item(), global_step)

        # backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # train accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} Loss {loss.item():.4f}")

    # ===== END OF BATCH LOOP =====

    # epoch metrics
    epoch_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    test_acc = evaluate(model, test_loader, device)

    # TensorBoard logging
    writer.add_scalar("Loss/epoch", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/test", test_acc, epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # scheduler.step()  # only if using SGD


# save model
torch.save(model.state_dict(), "resnet18_cifar10.pth")
print("Model saved successfully!")


# close TensorBoard writer
writer.close()