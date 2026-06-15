"""
Opti-DefectNet: UI Deployment
"""
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import sys
import os

# Append src to path to import the model class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import OptiNet

# Load trained weights
device = torch.device("cpu")
model = OptiNet()
try:
    model.load_state_dict(torch.load('models/opti_net_v1.pth', map_location=device))
    model.eval() # Set model to inference mode (disables dropout)
except FileNotFoundError:
    print("ERROR: Model weights not found. Run src/train.py first.")
    exit()

# Define identical inference transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_defect(image):
    # Convert Gradio image (numpy) to PIL, then to Tensor
    img = Image.fromarray(image).convert('RGB')
    input_tensor = transform(img).unsqueeze(0) # Add batch dimension

    with torch.no_grad(): # Disable gradient tracking for inference speed
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
    classes = ['Fail (Defect Detected)', 'Pass (Clean Surface)']
    # Return dictionary mapping class names to confidence percentages
    return {classes[i]: float(probabilities[i]) for i in range(2)}

# Construct Web Interface
interface = gr.Interface(
    fn=predict_defect,
    inputs=gr.Image(label="Upload Fiber Optic End-Face"),
    outputs=gr.Label(num_top_classes=2, label="Inspection Result"),
    title="Opti-DefectNet: Automated QA",
    description="Upload a microscopic image of a fiber optic end-face to detect scratches, pitting, or surface dust utilizing a custom PyTorch CNN."
)

if __name__ == "__main__":
    interface.launch(share=False)