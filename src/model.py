"""
Opti-DefectNet: CNN Architecture Definition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class OptiNet(nn.Module):
    def __init__(self):
        super(OptiNet, self).__init__()
        # Input: 3 channels (RGB), Output: 16 feature maps, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # Pooling cuts spatial dimensions in half (256x256 -> 128x128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Fully Connected (Dense) layers
        # After 3 max pools of 2x2, a 256x256 image becomes 32x32.
        # 64 channels * 32 * 32 = 65536 flattened tensor dimensions
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.dropout = nn.Dropout(p=0.5) # Regularization to prevent overfitting
        self.fc2 = nn.Linear(128, 2)     # 2 Output classes (Pass, Fail)

    def forward(self, x):
        # Forward pass: Conv -> ReLU (Activation) -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten tensor for the fully connected layers
        x = x.view(-1, 64 * 32 * 32)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Raw logits output (no softmax here, handled by CrossEntropyLoss)
        return x