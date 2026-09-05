"""
Opti-DefectNet: Standalone Model Evaluation

This script loads serialized model weights and runs inference against 
an evaluation dataset to compute and report overall classification accuracy.
"""
import torch
from torch.utils.data import DataLoader
from dataset import FiberDataset
from model import OptiNet

def evaluate_model():
    """
    Initializes the model architecture, loads trained weights, and processes 
    batches through an evaluation data loader to measure performance.
    """
    # Set the computation device.
    # Fine-tuning and Iteration: Change to cuda if evaluating on an NVIDIA GPU.
    device = torch.device("cpu")
    model = OptiNet()
    
    # Attempt to load serialized model weights from disk.
    try:
        model.load_state_dict(torch.load('models/opti_net_v1.pth', map_location=device))
    except FileNotFoundError:
        print("ERROR: Model weights not found. Please run src/train.py first.")
        return

    # Set the model to evaluation mode (disables dropout and freezes batch normalization).
    model.eval()
    
    # Load dataset in evaluation mode (is_train=False disables random augmentations).
    data_path = 'data/synthetic_generated'
    eval_dataset = FiberDataset(data_dir=data_path, is_train=False)
    
    # Construct a DataLoader to process samples in batches.
    # Fine-tuning and Iteration: Adjust batch_size depending on your system memory limits.
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0
    
    # Disable gradient calculation to conserve memory and accelerate evaluation speed.
    with torch.no_grad():
        for inputs, labels in eval_loader:
            outputs = model(inputs)
            
            # Extract the predicted class index (highest logit score across the 2 outputs).
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate and display overall model accuracy percentage.
    accuracy = 100 * correct / total
    print(f"Evaluated {total} images. Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()