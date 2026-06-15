"""
Opti-DefectNet: Synthetic Data Generator
Generates simulated fiber optic end-faces to train the CNN.
"""
import cv2
import numpy as np
import os
import random

def generate_fiber_image(is_defective, img_size=256):
    # Initialize a black canvas (sensor background)
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    center = (img_size // 2, img_size // 2)
    
    # Draw Cladding (Outer circle) - Dark Grey
    cv2.circle(img, center, radius=100, color=(80, 80, 80), thickness=-1)
    # Draw Core (Inner circle) - Lighter Grey/White
    cv2.circle(img, center, radius=20, color=(200, 200, 200), thickness=-1)
    
    # Simulate Gaussian noise (sensor static/imperfections)
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    if is_defective:
        # Inject procedural defects (scratches or pits)
        defect_type = random.choice(['scratch', 'pit'])
        if defect_type == 'scratch':
            # Generate 1 to 3 random lines across the end-face
            for _ in range(random.randint(1, 3)):
                x1, y1 = random.randint(50, 200), random.randint(50, 200)
                x2, y2 = random.randint(50, 200), random.randint(50, 200)
                cv2.line(img, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=random.randint(1, 3))
        else:
            # Generate microscopic dust/pits (small dark circles)
            for _ in range(random.randint(3, 10)):
                px, py = random.randint(80, 180), random.randint(80, 180)
                cv2.circle(img, (px, py), radius=random.randint(1, 4), color=(20, 20, 20), thickness=-1)
                
    # Apply slight blur to simulate focal depth constraints of microscopes
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def build_dataset(samples_per_class=1000, base_dir="data/synthetic_generated"):
    classes = {'pass': False, 'fail': True}
    for cls_name, is_defective in classes.items():
        dir_path = os.path.join(base_dir, cls_name)
        os.makedirs(dir_path, exist_ok=True)
        
        for i in range(samples_per_class):
            img = generate_fiber_image(is_defective)
            cv2.imwrite(os.path.join(dir_path, f"sample_{i:04d}.jpg"), img)
            
if __name__ == "__main__":
    print("Initializing synthetic data generation pipeline...")
    build_dataset(samples_per_class=1000) # Generates 2000 images total
    print("Dataset generation complete.")