import sys
import torch
from torchvision import models
from PIL import Image

# Check arguments
if len(sys.argv) != 2:
    print("Usage: uv run predict.py <path_to_image>")
    print("Example with image in this folder: uv run predict.py img.jpg")
    sys.exit(1)

image_path = sys.argv[1]

# Clases
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Model
model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
model.classifier[6] = torch.nn.Linear(4096, 6)
model.load_state_dict(torch.load("../../models/model_vgg11_garbage.pth", map_location="cpu"))
model.eval()

# Same transformations as in train.py
transform = models.VGG11_Weights.IMAGENET1K_V1.transforms()

# Load image
image = Image.open(image_path).convert("RGB")
tensor = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():
    outputs = model(tensor)
    pred_idx = outputs.argmax(1).item()
    pred_class = class_names[pred_idx]

print(f"Prediction: {pred_class} (class {pred_idx})")
