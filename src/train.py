"""
Opti-DefectNet: Training Pipeline with Validation

This script orchestrates the end-to-end model training loop, managing dataset splitting,
forward and backward optimization passes, validation evaluation, and weight serialization.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import FiberDataset
from model import OptiNet
import os

def train_model():
    """
    Executes the model training process across defined epochs, calculating loss,
    updating weights, tracking validation accuracy, and saving final checkpoints.
    """
    # Select hardware backend for tensor computations.
    # Fine-tuning and Iteration: Change to "cuda" if running on an NVIDIA GPU for accelerated training.
    device = torch.device("cpu")
    print(f"Executing on computational backend: {device}")

    # Load the full synthetic dataset using training augmentations.
    data_path = 'data/synthetic_generated'
    full_dataset = FiberDataset(data_dir=data_path, is_train=True)

    # Implement an 80/20 train-to-validation split to monitor generalization.
    # Fine-tuning and Iteration: Adjust percentage ratios (e.g., 0.85/0.15) if altering split sizes.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Wrap datasets into DataLoaders to process samples in mini-batches.
    # Fine-tuning and Iteration: Adjust batch_size (e.g., 16, 32, 64) based on system memory.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model instance, loss function, and optimizer.
    model = OptiNet().to(device)
    
    # CrossEntropyLoss combines LogSoftmax and NLLLoss, ideal for multi-class/binary classification logits.
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer adapts learning rates per parameter using first and second moment estimates.
    # Fine-tuning and Iteration: Adjust learning rate lr (e.g., 0.0001 to 0.01) if model fails to converge.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    # Fine-tuning and Iteration: Increase epochs (e.g., 10 or 20) for deeper convergence on complex datasets.
    for epoch in range(epochs):
        # Phase 1: Training Loop
        model.train() 
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clear previous step gradient computations.
            optimizer.zero_grad()
            
            # Forward pass: compute predicted logits.
            outputs = model(inputs)
            
            # Calculate classification loss.
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradients via backpropagation.
            loss.backward()
            
            # Update network weights.
            optimizer.step()
            
            running_train_loss += loss.item()
            
        # Phase 2: Validation Loop
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        
        # Disable gradient tracking during validation.
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                # Determine predicted class indices.
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Compute epoch summary metrics.
        avg_train_loss = running_train_loss / len(train_loader)
        avg_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

    # Save serialized model weights to disk after training completion.
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/opti_net_v1.pth')
    print("Model weights serialized and saved.")

if __name__ == "__main__":
    train_model()