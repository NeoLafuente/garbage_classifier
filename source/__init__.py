#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Garbage Classifier Project
===========================

A deep learning project for garbage classification using PyTorch Lightning
and transfer learning with ResNet18.

Overview
--------
This project implements a garbage classification system that categorizes waste
into 6 different classes: cardboard, glass, metal, paper, plastic, and trash.
The model uses a pretrained ResNet18 architecture fine-tuned on a custom
garbage dataset.

Project Structure
-----------------
- **train.py**: Main training script for the classification model
- **predict.py**: Script for making predictions on new images
- **utils/**: Utility modules containing configuration and custom classes
  - **config.py**: Configuration parameters and constants
  - **custom_classes/**: Custom PyTorch Lightning implementations
    - **GarbageClassifier.py**: ResNet18-based classifier model
    - **GarbageDataModule.py**: Data loading and preprocessing module
    - **LossCurveCallback.py**: Callback for visualizing training metrics

Features
--------
- Transfer learning with pretrained ResNet18 (ImageNet weights)
- Stratified train/test split (90/10) for balanced class distribution
- Automatic loss and accuracy curve generation
- GPU acceleration support with automatic fallback to CPU
- Command-line interface for training and prediction

Quick Start
-----------
**Training the model:**

    uv run train.py

**Making predictions:**

    uv run predict.py path/to/image.jpg

**Generating documentation:**

    uv run scripts/generate_docs.py

Dependencies
------------
- PyTorch Lightning: Deep learning framework
- PyTorch & torchvision: Neural network implementation and pretrained models
- PIL: Image processing
- scikit-learn: Dataset splitting utilities
- matplotlib: Visualization

Model Architecture
------------------
The classifier uses a ResNet18 architecture with:
- Pretrained feature extraction layers (frozen)
- Custom classification head for 6 garbage categories
- Cross-entropy loss function
- Adam optimizer

Dataset
-------
The model is trained on a custom garbage dataset with 6 classes:
1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash

Performance
-----------
Training metrics and performance visualizations are automatically saved to:
- Loss curves: `models/performance/loss_curves/`
- Model checkpoint: `models/weights/model_resnet18_garbage.ckpt`

Examples
--------
>>> # Import the classifier
>>> from utils.custom_classes.GarbageClassifier import GarbageClassifier
>>> from utils import config as cfg
>>>
>>> # Load trained model
>>> model = GarbageClassifier.load_from_checkpoint(
...     cfg.MODEL_PATH,
...     num_classes=cfg.NUM_CLASSES
... )
>>>
>>> # Make prediction
>>> from predict import predict_image
>>> pred_class, pred_idx = predict_image("sample.jpg")
>>> print(f"Prediction: {pred_class}")

Notes
-----
This project follows best practices for:
- Code organization and modularity
- Documentation (NumPy-style docstrings)
- Dependency management (using uv)
- Reproducibility (fixed random seeds in data splitting)

For more detailed information, see the individual module documentation.
"""

from source import train
from source import predict

__all__ = [
    "train",
    "predict",
    "utils",
]
