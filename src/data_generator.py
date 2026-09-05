"""
Opti-DefectNet: Synthetic Data Generator

This script utilizes OpenCV and NumPy to procedurally generate synthetic microscopic 
images of optical fiber end-faces, simulating both clean surfaces and defective surfaces 
(scratches and pits) to bootstrap the training pipeline.
"""
import cv2
import numpy as np
import os
import random

def generate_fiber_image(is_defective, img_size=256):
    """
    Constructs a single synthetic fiber end-face image with structural layers 
    and optional defect simulations.
    
    Fine-tuning and Iteration: Adjust radii, color intensity values, or noise parameters 
    to better match real-world microscopic camera captures.
    """
    # Initialize a blank black canvas image of specified dimensions with 3 color channels (RGB).
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    center = (img_size // 2, img_size // 2)
    
    # Draw the outer cladding layer of the optical fiber (dark gray circle).
    cv2.circle(img, center, radius=100, color=(80, 80, 80), thickness=-1)
    # Draw the inner core layer of the optical fiber (light gray circle).
    cv2.circle(img, center, radius=20, color=(200, 200, 200), thickness=-1)
    
    # Inject Gaussian sensor noise to simulate camera grain and texture.
    # Fine-tuning and Iteration: Increase the standard deviation parameter (e.g., from 15 to 30) 
    # to introduce heavier sensor noise.
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Conditionally inject structural defects if the sample is flagged as defective.
    if is_defective:
        defect_type = random.choice(['scratch', 'pit'])
        if defect_type == 'scratch':
            # Draw random dark lines simulating surface scratching.
            for _ in range(random.randint(1, 3)):
                x1, y1 = random.randint(50, 200), random.randint(50, 200)
                x2, y2 = random.randint(50, 200), random.randint(50, 200)
                cv2.line(img, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=random.randint(1, 3))
        else:
            # Draw random dark spots simulating pitting or debris contamination.
            for _ in range(random.randint(3, 10)):
                px, py = random.randint(80, 180), random.randint(80, 180)
                cv2.circle(img, (px, py), radius=random.randint(1, 4), color=(20, 20, 20), thickness=-1)
                
    # Apply a light Gaussian blur to soften harsh edges and blend defects naturally into the surface.
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def build_dataset(samples_per_class=1000, base_dir="data/synthetic_generated"):
    """
    Automates the mass generation and directory sorting of synthetic images 
    into 'pass' and 'fail' training folders.
    
    Fine-tuning and Iteration: Modify samples_per_class to scale dataset size up or down.
    """
    classes = {'pass': False, 'fail': True}
    for cls_name, is_defective in classes.items():
        dir_path = os.path.join(base_dir, cls_name)
        os.makedirs(dir_path, exist_ok=True)
        
        for i in range(samples_per_class):
            img = generate_fiber_image(is_defective)
            # Save each generated frame as a JPEG image file.
            cv2.imwrite(os.path.join(dir_path, f"sample_{i:04d}.jpg"), img)
            
if __name__ == "__main__":
    print("Initializing synthetic data generation pipeline...")
    build_dataset(samples_per_class=1000)
    print("Dataset generation complete.")