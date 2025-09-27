#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import torch
from torchvision import models, transforms
from PIL import Image
import pytorch_lightning as pl
from utils import config as cfg
from utils.custom_classes.GarbageClassifier import GarbageClassifier

if len(sys.argv) > 2:
    print("Usage: uv run predict.py <path_to_image>")
    print("Example with image in this folder: uv run predict.py img.jpg")
    sys.exit(1)
elif len(sys.argv) == 1:
    image_path = cfg.SAMPLE_IMG_PATH
else:
    image_path = sys.argv[1]

class_names = cfg.CLASS_NAMES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print(f"Loading model...")
model = GarbageClassifier.load_from_checkpoint(cfg.MODEL_PATH, num_classes = cfg.NUM_CLASSES)

model = model.to(device)
model.eval()

print(f"Transforming image...")
transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
image = Image.open(image_path).convert("RGB")
tensor = transform(image).unsqueeze(0).to(device)


with torch.no_grad():
    outputs = model(tensor)
    pred_idx = outputs.argmax(1).item()
    pred_class = class_names[pred_idx]

print(f"Prediction: {pred_class} (class {pred_idx})")
