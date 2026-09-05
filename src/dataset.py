"""
Opti-DefectNet: Custom Dataset Pipeline

This module subclasses PyTorch's Dataset class to load image file paths, 
assign binary classification labels, and apply real-time data augmentations 
to train robust deep learning models.
"""
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FiberDataset(Dataset):
    """
    Custom PyTorch Dataset for loading, parsing, and transforming 
    optical fiber end-face inspection images.
    """
    def __init__(self, data_dir, is_train=True):
        """
        Initializes the dataset by scanning directory paths and configuring 
        appropriate transformation pipelines based on training versus validation mode.
        
        Fine-tuning and Iteration: Add custom folder mappings or classification 
        indices here if expanding to multi-class defect categories (e.g., scratches vs. dust).
        """
        self.data_dir = data_dir
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        
        # Map target directory folder names to integer class labels.
        # pass = 0 (Clean), fail = 1 (Defective)
        self.class_to_idx = {'pass': 0, 'fail': 1}
        
        # Traverse subdirectories to compile file paths and label lists.
        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(idx)
                        
        # Configure image augmentation pipelines.
        if self.is_train:
            # Training pipeline incorporates random perturbations to prevent model memorization.
            # Fine-tuning and Iteration: Adjust rotation degrees or add color jitter 
            # to simulate harsher industrial lighting variations.
            self.transform = transforms.Compose([
                # Resize is redundant for 256x256 synthetic data, but intentionally 
                # included to standardize varying resolutions of real-world manufacturing images.
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15), 
                transforms.GaussianBlur(kernel_size=3), 
                transforms.ToTensor(),
                # Normalize pixel intensity values to a [-1, 1] scale.
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            # Validation pipeline strictly standardizes images without random modifications.
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        """Returns the total number of samples available in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches, opens, transforms, and returns a single image tensor and its corresponding label by index.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image via PIL and ensure 3-channel RGB formatting.
        image = Image.open(img_path).convert("RGB")
        
        # Apply the configured transformation pipeline.
        if self.transform:
            image = self.transform(image)
            
        return image, label