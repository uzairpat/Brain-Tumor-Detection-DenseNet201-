import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import os

# Define dataset paths
train_dir = '../brain_tumor/input/Training'
test_dir = '../brain_tumor/input/Testing'

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(root=train_dir, transform=train_transform)
test_dataset = ImageFolder(root=test_dir, transform=test_transform)

# Check if datasets are loaded correctly
print(f"Training dataset size: {len(train_dataset)}")
print(f"Testing dataset size: {len(test_dataset)}")
if len(train_dataset) == 0 or len(test_dataset) == 0:
    raise RuntimeError("Dataset is empty. Check the dataset path.")

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Get class names
class_names = train_dataset.classes
print(f"Classes: {class_names}")

# Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from torchvision.models import DenseNet201_Weights
model = models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(class_names))
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=10, target_accuracy=99.0):
    train_losses = []
    val_accuracies = []
    total_batches = len(train_loader)
    print(f"Total batches per epoch: {total_batches}")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} is running... (Learning Rate: {optimizer.param_groups[0]['lr']})")
        model.train()
        running_loss = 0.0
        batch_count = 0
        for images, labels in train_loader:
            batch_count += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Epoch {epoch+1}, Batch {batch_count}/{total_batches}, Loss: {running_loss / batch_count:.4f}")
        train_losses.append(running_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {train_losses[-1]:.4f}")
    return train_losses

# Train model
print("Starting training...")
train_losses = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=10)
print("Training completed.")

# Save model
save_path = '../brain_tumor/densenet201_brain_tumor.pth'
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
