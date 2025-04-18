import torch
import torch.nn as nn

class BloodGroupCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(BloodGroupCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def load_model(model_path):
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Initialize the model
    model = BloodGroupCNN(num_classes=len(checkpoint['class_names']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Return the model and class names
    return model, checkpoint['class_names']

