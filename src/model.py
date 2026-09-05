"""
Opti-DefectNet: CNN Architecture Definition

This module defines the custom 4-layer Convolutional Neural Network (CNN) architecture 
designed for binary optical fiber end-face classification (Pass vs. Fail).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class OptiNet(nn.Module):
    """
    A custom 4-stage convolutional neural network featuring batch normalization,
    max pooling, dropout regularization, and fully connected classification layers.
    """
    def __init__(self):
        """
        Initializes network layers, filter depths, kernel sizes, and linear transformations.
        
        Fine-tuning and Iteration: Adjust out_channels (e.g., 16 -> 32) or add additional 
        convolutional blocks here if scaling model capacity for higher-resolution images.
        """
        super(OptiNet, self).__init__()
        
        # Layer 1: Accepts 3-channel RGB images, outputs 16 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Layer 2: 16 -> 32 feature maps
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Layer 3: 32 -> 64 feature maps
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Layer 4: 64 -> 128 feature maps
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Max pooling cuts spatial dimensions in half at each stage (2x2 kernel, stride 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected (Dense) layers
        # Spatial dimension progression from 256x256 input across 4 pooling steps:
        # 256 -> 128 -> 64 -> 32 -> 16. Flattened tensor size = 128 channels * 16 * 16 = 32,768.
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        
        # Dropout regularization randomly zeroes 50% of node activations during training to prevent overfitting.
        # Fine-tuning and Iteration: Adjust probability p (e.g., 0.3 to 0.6) based on training/validation loss gaps.
        self.dropout = nn.Dropout(p=0.5) 
        
        # Output classification head mapping to 2 output classes (Pass, Fail)
        self.fc2 = nn.Linear(128, 2)     

    def forward(self, x):
        """
        Defines the forward pass execution flow: Convolution -> Batch Norm -> ReLU -> Max Pool.
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten the 3D tensor into a 2D batch tensor for the dense layers.
        x = x.view(-1, 128 * 16 * 16)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Returns raw unnormalized logits (CrossEntropyLoss handles softmax internally).
        return x