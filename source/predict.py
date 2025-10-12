#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Garbage Classification Prediction Script.

This script loads a trained GarbageClassifier model and performs inference
on images to predict their garbage category. The script can accept either
a single image path or a directory path as a command-line argument, or use
a default sample image.

The prediction uses a pretrained ResNet18 model fine-tuned for 6-class
garbage classification (cardboard, glass, metal, paper, plastic, trash).

Usage
-----
Command line:

    $ uv run predict.py <path_to_image_or_folder>

Examples
--------
Predict with custom image:

    $ uv run predict.py img.jpg

Predict with folder:

    $ uv run predict.py /path/to/images/

Predict with default sample image:

    $ uv run predict.py

Notes
-----
- The model checkpoint path is configured in `utils.config`
- Images are automatically preprocessed using ImageNet normalization
- Prediction runs on GPU if available, otherwise falls back to CPU
- When processing folders, only common image formats are considered
  (jpg, jpeg, png, bmp, gif, tiff)
"""
__docformat__ = "numpy"

import sys
from pathlib import Path
import torch
from torchvision import models
from PIL import Image
from utils import config as cfg
from utils.custom_classes.GarbageClassifier import GarbageClassifier


def predict_image(image_path, model, transform, device, class_names):
    """
    Predict the garbage category of an input image.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file to classify.
    model : GarbageClassifier
        The loaded model for inference.
    transform : torchvision.transforms.Compose
        Image transformation pipeline to apply before inference.
    device : torch.device
        Device (CPU or GPU) where tensors will be allocated.
    class_names : list of str
        List of class names corresponding to model outputs.

    Returns
    -------
    tuple of (str, int)
        A tuple containing the predicted class name and class index.

    Examples
    --------
    >>> pred_class, pred_idx = predict_image(
            "sample.jpg", model, transform, device, class_names
        )
    >>> print(f"Prediction: {pred_class}")

    Notes
    -----
    The function applies the appropriate image transformations for the
    ResNet18 model and automatically handles RGB conversion for images
    with different color modes.
    """
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        pred_idx = outputs.argmax(1).item()
        pred_class = class_names[pred_idx]

    return pred_class, pred_idx


def get_image_files(path):
    """
    Get all valid image files from a directory.

    Parameters
    ----------
    path : Path
        Path to the directory to search for images.

    Returns
    -------
    list of Path
        Sorted list of Path objects pointing to valid image files.

    Notes
    -----
    Supported image formats: jpg, jpeg, png, bmp, gif, tiff, tif
    Files are returned in sorted order for consistent processing.
    """
    valid_extensions = {
        '.jpg',
        '.jpeg',
        '.png',
        '.bmp',
        '.gif',
        '.tiff',
        '.tif'
    }
    image_files = [
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    return sorted(image_files)


def predict_folder(folder_path):
    """
    Predict the garbage category for all images in a folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to the folder containing images to classify.

    Notes
    -----
    - The model is loaded once and reused for all images for efficiency
    - Progress is displayed for each processed image
    - Invalid images are skipped with a warning message
    - A summary with all predictions is displayed at the end
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)

    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        sys.exit(1)

    image_files = get_image_files(folder)

    if not image_files:
        print(f"No valid image files found in '{folder_path}'")
        sys.exit(1)

    print(f"Found {len(image_files)} image(s) to process\n")

    # Setup device and model
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

    print("Preparing transformations...\n")
    transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()

    # Process all images
    results = []
    for idx, image_path in enumerate(image_files, 1):
        try:
            print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")
            pred_class, pred_idx = predict_image(
                image_path, model, transform, device, class_names
            )
            print(f"  -> Prediction: {pred_class} (class {pred_idx})\n")
            results.append((image_path.name, pred_class, pred_idx))
        except Exception as e:
            print(f"  -> Error processing {image_path.name}: {str(e)}\n")
            continue

    # Print summary
    if results:
        print("=" * 60)
        print("PREDICTION SUMMARY")
        print("=" * 60)
        for filename, pred_class, pred_idx in results:
            print(f"{filename:<40} -> {pred_class} (class {pred_idx})")
        print("=" * 60)


def predict_single_image(image_path):
    """
    Predict the garbage category for a single image.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file to classify.

    Notes
    -----
    This function handles the complete prediction pipeline for a single
    image, including model loading, device selection, and result display.
    """
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

    pred_class, pred_idx = predict_image(
        image_path, model, transform, device, class_names
    )
    print(f"Prediction: {pred_class} (class {pred_idx})")


def main():
    """
    Main entry point for the prediction script.

    Parses command-line arguments and performs prediction on the specified
    image, folder, or a default sample image. Automatically detects whether
    the input is a file or directory and processes accordingly.

    Notes
    -----
    When a directory is provided, all valid image files within it are
    processed sequentially using the same loaded model for efficiency.
    """
    if len(sys.argv) > 2:
        print("Usage: uv run predict.py <path_to_image_or_folder>")
        print("Examples:")
        print("  uv run predict.py img.jpg")
        print("  uv run predict.py /path/to/images/")
        sys.exit(1)
    elif len(sys.argv) == 1:
        image_path = cfg.SAMPLE_IMG_PATH
        predict_single_image(image_path)
    else:
        input_path = Path(sys.argv[1])

        if not input_path.exists():
            print(f"Error: Path '{input_path}' does not exist.")
            sys.exit(1)

        if input_path.is_dir():
            predict_folder(input_path)
        else:
            predict_single_image(input_path)


if __name__ == "__main__":
    main()
