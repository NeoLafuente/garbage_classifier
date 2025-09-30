#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__docformat__ = "numpy"

"""
Garbage Classification Prediction Script.

This script loads a trained GarbageClassifier model and performs inference
on a single image to predict its garbage category. The script can accept
an image path as a command-line argument or use a default sample image.

The prediction uses a pretrained ResNet18 model fine-tuned for 6-class
garbage classification (cardboard, glass, metal, paper, plastic, trash).

Usage
-----
Command line::

    uv run predict.py <path_to_image>

Examples
--------
Predict with custom image::

    uv run predict.py img.jpg

Predict with default sample image::

    uv run predict.py

Notes
-----
- The model checkpoint path is configured in `utils.config`
- Images are automatically preprocessed using ImageNet normalization
- Prediction runs on GPU if available, otherwise falls back to CPU
"""

import sys
import torch
from torchvision import models
from PIL import Image
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

print("Loading model...")
model = GarbageClassifier.load_from_checkpoint(
    cfg.MODEL_PATH,
    num_classes=cfg.NUM_CLASSES
)

model = model.to(device)
model.eval()

print("Transforming image...")
transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
image = Image.open(image_path).convert("RGB")
tensor = transform(image).unsqueeze(0).to(device)


with torch.no_grad():
    outputs = model(tensor)
    pred_idx = outputs.argmax(1).item()
    pred_class = class_names[pred_idx]

print(f"Prediction: {pred_class} (class {pred_idx})")
