"""
Opti-DefectNet: UI Deployment with Built-in Feedback Logging

This script launches a Gradio web application that loads a trained PyTorch model
to perform real-time optical fiber end-face quality assurance inspections and 
logs user feedback for model iteration.
"""
import os
import csv
from datetime import datetime
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import sys

# Add the 'src' directory to the Python path so local modules like model.py can be imported.
# Fine-tuning and Iteration: Adjust the path if you reorganize your directory structure.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import OptiNet

# Define the compute device for inference.
# Fine-tuning and Iteration: Change to cuda if running on an NVIDIA GPU, or mps for Apple Silicon.
device = torch.device("cpu")
model = OptiNet()

# Load the serialized model weights from training.
# Fine-tuning and Iteration: Update the file path if you train a newer model version such as opti_net_v2.pth.
model.load_state_dict(torch.load('models/opti_net_v1.pth', map_location=device))
model.eval() # Set the model to evaluation mode to disable dropout and batch normalization updates.

# Define the input preprocessing pipeline for inference.
# Fine-tuning and Iteration: Ensure these transformations match the validation and test transforms in dataset.py.
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_defect(image):
    """
    Takes an uploaded image numpy array, preprocesses it, runs model inference,
    and returns a dictionary of class probabilities.
    """
    if image is None:
        return {"Error: Please upload an image": 1.0}
        
    # Convert NumPy array from Gradio into a PIL Image and ensure RGB format.
    img = Image.fromarray(image).convert('RGB')
    
    # Apply transforms and add a batch dimension of shape [1, 3, 256, 256].
    input_tensor = transform(img).unsqueeze(0) 

    # Disable gradient tracking during inference to reduce memory usage and speed up execution.
    with torch.no_grad(): 
        output = model(input_tensor)
        # Apply softmax to raw logits to obtain percentage probabilities.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    # Map model output indices to readable category names.
    # Fine-tuning and Iteration: Ensure the class order matches your dataset directory structure index mapping.
    classes = ['Pass (Clean Surface)', 'Fail (Defect Detected)']
    return {classes[i]: float(probabilities[i]) for i in range(2)}

def log_feedback(image, prediction, user_tag):
    """
    Manually logs user corrections, automated model predictions, confidence scores,
    and saves flagged images to a structured local directory and human-readable CSV.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    model_pred = "Unknown"
    confidence = "0.0%"
    if isinstance(prediction, dict):
        model_pred = max(prediction, key=prediction.get)
        confidence = f"{prediction[model_pred] * 100:.2f}%"

    correction = user_tag if user_tag else "None (Automated Flag)"

    # Set up local storage directories for feedback logging.
    # Fine-tuning and Iteration: Change flagged_feedback to a cloud bucket directory if scaling to production.
    flagging_dir = "flagged_feedback"
    img_folder = os.path.join(flagging_dir, "images")
    os.makedirs(img_folder, exist_ok=True)
    
    image_name = f"flag_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    img_path = os.path.join(img_folder, image_name)
    
    # Save the uploaded image locally for future dataset retraining iterations.
    if image is not None:
        try:
            Image.fromarray(image).save(img_path)
        except Exception:
            pass

    # Append structured metadata to a human-readable CSV log.
    csv_path = os.path.join(flagging_dir, "curated_feedback_log.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header row only if the file is newly created.
        if not file_exists:
            writer.writerow([
                "Timestamp", 
                "Image File", 
                "Automated Model Prediction", 
                "Model Confidence", 
                "User Correction / Flag Tag"
            ])
        writer.writerow([
            timestamp, 
            img_path, 
            model_pred, 
            confidence, 
            correction
        ])
    return "Feedback successfully logged to curated_feedback_log.csv."

# Build the Gradio web interface layout using Blocks.
# Fine-tuning and Iteration: Add CSS themes or custom branding elements here if customizing the UI.
with gr.Blocks(title="Opti-DefectNet: Automated QA") as demo:
    gr.Markdown("# Opti-DefectNet: Automated QA")
    gr.Markdown("Upload a microscopic image to detect optical anomalies and provide corrective feedback.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Fiber Optic End-Face")
            submit_btn = gr.Button("Run Inspection", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=2, label="Inspection Result")
            
    # Bind the prediction function to the submit button click event.
    submit_btn.click(fn=predict_defect, inputs=image_input, outputs=label_output)
    
    gr.Markdown("### Log Feedback & Corrections")
    with gr.Row():
        correction_dropdown = gr.Dropdown(
            choices=["Pass (Clean Surface)", "Fail (Defect Detected)"], 
            label="Correct Category"
        )
        flag_btn = gr.Button("Submit Feedback")
    
    status_output = gr.Textbox(label="Status Log", interactive=False)
    
    # Bind the feedback logging function to the feedback submission button click event.
    flag_btn.click(
        fn=log_feedback, 
        inputs=[image_input, label_output, correction_dropdown], 
        outputs=status_output
    )

if __name__ == "__main__":
    # Launch the local web server.
    # Fine-tuning and Iteration: Set share=True to generate a public URL for remote testing.
    demo.launch(share=False)