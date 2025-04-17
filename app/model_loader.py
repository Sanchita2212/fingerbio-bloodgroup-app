import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# Resolve class names path (relative to this file)
CLASS_NAMES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'model', 'class_names.txt')
)

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# CNN model definition
class BloodGroupCNN(nn.Module):
    def __init__(self, num_classes=8, class_names=None):
        super(BloodGroupCNN, self).__init__()
        self.class_names = class_names
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

# Load model
def load_model(model_path='blood_group_model_best.pth'):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create the model and pass class_names to it
    model = BloodGroupCNN(num_classes=len(checkpoint['class_names']), class_names=checkpoint['class_names'])
    
    # Load the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()  # Set model to evaluation mode
    return model

# Predict image
def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return model.class_names[predicted.item()]  # Access class_names from model


