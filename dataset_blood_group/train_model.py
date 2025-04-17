import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import Counter

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
data_dir = 'C:/Users/ankit/Downloads/fingerprint/dataset_blood_group'
model_path = 'blood_group_model_best.pth'

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset
full_data = datasets.ImageFolder(root=data_dir)
class_names = full_data.classes

# Class weights
labels = [label for _, label in full_data]
label_counts = Counter(labels)
total = sum(label_counts.values())
class_weights = [total / label_counts[i] for i in range(len(class_names))]
weights_tensor = torch.FloatTensor(class_weights).to(device)

# Split dataset
train_size = int(0.8 * len(full_data))
test_size = len(full_data) - train_size
train_data, test_data = random_split(full_data, [train_size, test_size], generator=torch.Generator().manual_seed(42))
train_data.dataset.transform = train_transform
test_data.dataset.transform = test_transform

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Model architecture
class BloodGroupCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(BloodGroupCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Initialize model
model = BloodGroupCNN(num_classes=len(class_names)).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training loop
num_epochs = 10
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {running_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': class_names
        }, model_path)

    scheduler.step()

print(f"\n✅ Best Validation Accuracy: {best_acc:.2f}%")
print(f"✅ Model saved to {model_path}")
