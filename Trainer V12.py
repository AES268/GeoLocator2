import torch
import torchvision.transforms as transforms
import torchvision.datasets
import torchvision.models
from torch import nn, optim
from tqdm import tqdm

# Settings
data_dir = 'D:\Image Stuff\DATASET'
num_classes = 4
batch_size = 12
epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Datasets
class_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transforms)
class_loader = torch.utils.data.DataLoader(class_dataset, batch_size=batch_size, shuffle=True)

# Model (pretrained)
model = torchvision.models.resnet50(pretrained=True)

# Training loop with progress bar
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

    print(f"Epoch {epoch + 1} completed.")
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth!")

print("Training done!")