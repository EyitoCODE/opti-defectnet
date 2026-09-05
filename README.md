# Opti-DefectNet: Fiber Surface Inspection AI

## Overview
Opti-DefectNet is an end-to-end Machine Learning Operations (MLOps) pipeline designed to automate the quality assurance of optical fiber end-faces. Built from scratch using PyTorch and OpenCV, the system leverages a custom Convolutional Neural Network (CNN) to detect microscopic surface defects—such as scratches, pits, and debris contamination—classifying components as manufacturing-viable or defective. It features a complete workflow including procedural data generation, model training, performance evaluation, and an interactive Gradio web application with manual human-in-the-loop feedback logging.

---

## Project Structure & Architecture
The repository is organized into modular directories to separate core logic, dataset assets, model weights, and deployment interfaces:

```text
opti-defectnet/
├── app/
│   └── main.py                 # Gradio web interface & interactive feedback logging
├── data/
│   └── synthetic_generated/    # Procedurally generated training images (Pass / Fail)
├── models/
│   └── opti_net_v1.pth         # Serialized model weights generated after training
├── src/
│   ├── data_generator.py       # OpenCV script for generating synthetic fiber imagery
│   ├── dataset.py              # PyTorch Dataset subclass with real-time data augmentation
│   ├── model.py                # Custom 4-layer CNN architecture definition
│   ├── train.py                # Training script with 80/20 train/validation split
│   └── evaluate.py             # Standalone accuracy testing script
├── flagged_feedback/           # Runtime UI feedback logs and corrected images
│   ├── images/                 # Saved image captures from user submissions
│   └── curated_feedback_log.csv# Human-readable audit log of predictions vs. user corrections
├── .gitignore                  # Excludes virtual environments, weights, datasets, and logs
├── README.md                   # Comprehensive system documentation
└── requirements.txt            # Project Python dependencies
```

## Technical Implementation
* **Architecture**:  A custom 4-stage convolutional neural network (OptiNet) featuring progressive feature map expansion (16 -> 32 -> 64 -> 128 channels), Batch Normalization for gradient stability, 2x2 Max Pooling for spatial downsampling, and a Dropout layer ($p=0.5$) in the fully connected dense layers to prevent overfitting.

* **Data Pipeline**: A custom PyTorch Dataset pipeline that ingests 2,000 synthetically generated images (1,000 pass, 1,000 fail) and applies real-time data augmentations (random horizontal/vertical flips, rotations up to 15 degrees, Gaussian blur, and tensor normalization across RGB channels).

* **Training & Optimization:**: Managed via train.py utilizing the Adam optimizer ($\text{lr} = 0.001$), Cross-Entropy Loss, and an 80/20 stratified split.

## Quick Start
Follow these steps to clone the repository, generate data, train the model, evaluate performance, and launch the user interface from scratch.

```bash
# 1. Clone the repository and navigate into it
git clone https://github.com/EyitoCODE/opti-defectnet.git
cd opti-defectnet

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate  # If on Windows 
source .venv/bin/activate  # If on Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the synthetic dataset
python src/data_generator.py

# Fine-tuning: Open src/data_generator.py and adjust samples_per_class or modify the OpenCV drawing loop parameters to simulate different scratch sizes or noise densities.

# 5. Train the CNN model.
python src/train.py

# Output: Successfully saves trained weights to models/opti_net_v1.pth.

# Fine-tuning: Open src/train.py to adjust hyper-parameters such as learning rate (lr), batch size, or total epochs.

# 6. Evaluate Model Accuracy
python src/evaluate.py

# 7. Launch the Gradio Web UI
python app/main.py

```
## Accessing Important Data & Feedback Logs
The application implements an active MLOps feedback loop. When deployed in a manufacturing setting or reviewed during an evaluation, users can submit corrections for misclassified samples directly through the web interface.

* **CSV Schema**:
```text
# 1. Timestamp: Exact date and time of the inspection
# 2. Image File: Relative file path to the saved image capture.
# 3. Automated Model Prediction: The class predicted by OptiNet (Pass vs. Fail).
# 4. Model Confidence: Percentage confidence score of the prediction.
# 5. User Correction / Flag Tag: The manual category tag selected by the quality inspector.
```

* **Audit Logs**: All user interactions, model prediction confidence scores, and correction tags are logged automatically into a clean, human-readable CSV file located at:flagged_feedback/curated_feedback_log.csv

* **Archived Imagery**: High-resolution copies of flagged or corrected end-face images are archived inside: flagged_feedback/images/

* **Retraining Integration**: The data stored in flagged_feedback/images/ and audited via curated_feedback_log.csv can be ingested directly into subsequent training pipelines to continuously fine-tune model weights against edge cases encountered in production.