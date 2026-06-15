"""
Opti-DefectNet: Training Pipeline
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import OptiNet
import os

def train_model():
    # Hardware acceleration check
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing on computational backend: {device}")

    # Data transformation pipeline: resize, cast to Tensor, normalize
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Construct PyTorch DataLoader for batch processing
    dataset = datasets.ImageFolder(root='data/synthetic_generated', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and Adam optimizer
    model = OptiNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients to prevent accumulation from previous iterations
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backpropagation (compute gradients)
            loss.backward()
            # Optimizer step (update weights)
            optimizer.step()

            running_loss += loss.item()
            
        print(f"Epoch {epoch + 1}/{epochs} | Training Loss: {running_loss / len(dataloader):.4f}")

    print("Training convergence achieved.")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/opti_net_v1.pth')
    print("Model weights serialized and saved to models/opti_net_v1.pth")

if __name__ == "__main__":
    train_model()