# source/predict.py
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Garbage Classification Prediction Script.
...
"""
__docformat__ = "numpy"

import sys
from pathlib import Path
import torch
from torchvision import models
from PIL import Image
from source.utils import config as cfg
from source.utils.custom_classes.GarbageClassifier import GarbageClassifier
from codecarbon import EmissionsTracker
from source.utils.config import get_valid_dir as gvd

# ========================
# CORE PREDICTION FUNCTIONS (importable)
# ========================


def load_model_for_inference(model_path=None, device=None):
    """
    Load a trained model and preprocessing pipeline for inference.

    Loads a GarbageClassifier model from checkpoint and prepares it for
    inference. Automatically selects GPU if available, otherwise uses CPU.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to model checkpoint file. If None, uses cfg.MODEL_PATH.
        Default is None.
    device : torch.device, optional
        Device to load model on (CPU or CUDA). If None, auto-detects GPU
        availability. Default is None.

    Returns
    -------
    tuple
        A tuple containing:
        - model : GarbageClassifier
            Loaded model in evaluation mode.
        - device : torch.device
            Device the model is loaded on.
        - transform : torchvision.transforms.Compose
            Image preprocessing pipeline (ResNet18 ImageNet normalization).

    Examples
    --------
    >>> model, device, transform = load_model_for_inference()
    >>> # Use in Gradio app or API
    """
    if model_path is None:
        model_path = cfg.ensure_model_downloaded()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GarbageClassifier.load_from_checkpoint(
        str(model_path), num_classes=cfg.NUM_CLASSES
    )
    model = model.to(device)
    model.eval()

    transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()

    return model, device, transform


def predict_image(
    image_path,
    model=None,
    transform=None,
    device=None,
    class_names=None,
    track_carbon=False,
):
    """
    Predict the garbage category of a single image.

    Loads image from file or PIL Image object, applies preprocessing, and
    returns predictions with confidence scores for all classes. Optionally
    tracks carbon emissions for the inference operation.

    Parameters
    ----------
    image_path : str, Path, or PIL.Image
        Path to image file (str or Path object) or PIL Image object directly.
        Supported formats: JPG, PNG, BMP, GIF, TIFF.
    model : GarbageClassifier, optional
        Pre-loaded model. If None, loads from cfg.MODEL_PATH. Default is None.
    transform : torchvision.transforms.Compose, optional
        Image preprocessing pipeline. If None, uses default ResNet18's.
        Default is None.
    device : torch.device, optional
        Device for inference. If None, auto-selects GPU/CPU. Default is None.
    class_names : list of str, optional
        List of class names. If None, uses cfg.CLASS_NAMES. Default is None.
    track_carbon : bool, default=False
        Whether to track carbon emissions for this inference operation.

    Returns
    -------
    dict
        Dictionary containing:
        - 'predicted_class': str
            Predicted garbage class name.
        - 'predicted_idx': int
            Predicted class index (0 to num_classes-1).
        - 'confidence': float
            Confidence score for predicted class (0.0 to 1.0).
        - 'probabilities': dict
            All class probabilities {class_name: score, ...}.
        - 'emissions': dict or None
            Carbon emissions data if track_carbon=True, else None.
        - 'image': PIL.Image
            Original input image (RGB format).

    Raises
    ------
    FileNotFoundError
        If image file path does not exist.
    IOError
        If image file cannot be read.
    Exception
        If image format is not supported.

    Examples
    --------
    >>> result = predict_image("sample.jpg", track_carbon=True)
    >>> print(f"Prediction: {result['predicted_class']}")
    >>> print(f"Confidence: {result['confidence']:.2%}")
    """
    # Load model if not provided
    if model is None or transform is None or device is None:
        model, device, transform = load_model_for_inference(device=device)

    if class_names is None:
        class_names = cfg.CLASS_NAMES

    # Start carbon tracking if enabled
    emissions_data = None
    if track_carbon:
        tracker = EmissionsTracker(
            project_name="garbage_classifier_inference",
            output_dir=gvd(cfg.BEST_MODEL_DIR),
            log_level="warning",
        )
        tracker.start()

    try:
        # Handle both file paths and PIL Images
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            image = image_path.convert("RGB")
        else:
            # Assume numpy array (from Gradio)
            image = Image.fromarray(image_path).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = outputs.argmax(1).item()
            pred_class = class_names[pred_idx]
            confidence = probs[pred_idx].item()

        # Get all probabilities
        all_probs = {
            class_names[i]: probs[i].item() for i in range(len(class_names))
        }

        # Stop carbon tracking
        if track_carbon:
            emissions_kg = tracker.stop()
            from source.utils.carbon_utils import (
                kg_co2_to_car_distance,
                format_car_distance,
            )

            car_distances = kg_co2_to_car_distance(emissions_kg)
            emissions_data = {
                "emissions_kg": emissions_kg,
                "emissions_g": emissions_kg * 1000,
                "car_distance_km": car_distances["distance_km"],
                "car_distance_m": car_distances["distance_m"],
                "car_distance_formatted": format_car_distance(emissions_kg),
            }

        return {
            "predicted_class": pred_class,
            "predicted_idx": pred_idx,
            "confidence": confidence,
            "probabilities": all_probs,
            "emissions": emissions_data,
            "image": image,  # Return PIL image for display
        }

    except Exception as e:
        if track_carbon:
            tracker.stop()
        raise e


def predict_batch(
    image_paths,
    model=None,
    transform=None,
    device=None,
    class_names=None,
    track_carbon=False,
    progress_callback=None,
):
    """
    Predict garbage categories for multiple images.

    Processes a list of images efficiently using a single loaded model.
    Optionally tracks carbon emissions for the entire batch. Supports progress
    callbacks for UI integration.

    Parameters
    ----------
    image_paths : list of (str or Path)
        List of image file paths to process.
    model : GarbageClassifier, optional
        Pre-loaded model. If None, loads from cfg.MODEL_PATH. Default is None.
    transform : torchvision.transforms.Compose, optional
        Image preprocessing pipeline. Default is None.
    device : torch.device, optional
        Device for inference. Default is None (auto-detect).
    class_names : list of str, optional
        List of class names. Default is None (uses cfg.CLASS_NAMES).
    track_carbon : bool, default=False
        Whether to track carbon emissions for the entire batch.
    progress_callback : callable, optional
        Callback function called as progress_callback(current, total, message)
        where current is the image index (1-indexed), total is the total count,
        and message describes the current operation. Default is None.

    Returns
    -------
    dict
        Dictionary containing:
        - 'results': list of dict
            List of prediction results (one per image) with keys:
            'filename', 'predicted_class', 'predicted_idx', 'confidence',
            'probabilities', 'status' ('success' / 'error'), [and 'error'].
        - 'summary': dict
            Statistics with keys: 'total_images', 'successful', 'failed'.
        - 'emissions': dict or None
            Carbon emissions data with 'emissions_per_image_g' if tracked,
            else None.

    Notes
    -----
    Failed predictions are recorded with status='error' and error message.
    Model is loaded only once for efficiency. Individual image
    inference is not trackedfor carbon (only the batch total)
    to avoid overhead.

    Examples
    --------
    >>> results = predict_batch(["img1.jpg", "img2.jpg"], track_carbon=True)
    >>> for r in results['results']:
    ...     if r['status'] == 'success':
    ...         print(f"{r['filename']}: {r['predicted_class']}")
    """
    # Load model once for all images
    if model is None or transform is None or device is None:
        model, device, transform = load_model_for_inference(device=device)

    if class_names is None:
        class_names = cfg.CLASS_NAMES

    # Start carbon tracking if enabled
    emissions_data = None
    if track_carbon:
        tracker = EmissionsTracker(
            project_name="garbage_classifier_batch_inference",
            output_dir=gvd(cfg.BEST_MODEL_DIR),
            log_level="warning",
        )
        tracker.start()

    results = []
    total = len(image_paths)

    for idx, image_path in enumerate(image_paths):
        if progress_callback:
            progress_callback(
                idx + 1, total,
                f"Processing {Path(image_path).name}"
            )

        try:
            result = predict_image(
                image_path,
                model=model,
                transform=transform,
                device=device,
                class_names=class_names,
                track_carbon=False,
            )
            result["filename"] = Path(image_path).name
            result["status"] = "success"
            results.append(result)

        except Exception as e:
            results.append(
                {
                    "filename": Path(image_path).name,
                    "status": "error", "error": str(e)
                }
            )

    # Stop carbon tracking
    if track_carbon:
        emissions_kg = tracker.stop()
        from source.utils.carbon_utils import (
            kg_co2_to_car_distance,
            format_car_distance,
        )

        car_distances = kg_co2_to_car_distance(emissions_kg)
        emissions_data = {
            "emissions_kg": emissions_kg,
            "emissions_g": emissions_kg * 1000,
            "car_distance_km": car_distances["distance_km"],
            "car_distance_m": car_distances["distance_m"],
            "car_distance_formatted": format_car_distance(emissions_kg),
            "emissions_per_image_g": (emissions_kg * 1000) / len(image_paths),
        }

    # Summary
    successful = len([r for r in results if r.get("status") == "success"])
    summary = {
        "total_images": total,
        "successful": successful,
        "failed": total - successful,
    }

    final_result = {
        "results": results,
        "summary": summary,
        "emissions": emissions_data
    }

    return final_result


def get_image_files(path):
    """
    Get all valid image files from a directory.

    Recursively searches a directory for image files with supported extensions.
    Returns sorted list of image file paths.

    Parameters
    ----------
    path : Path or str
        Directory path to search for image files.

    Returns
    -------
    list of Path
        Sorted list of image file paths. Supported extensions: .jpg, .jpeg,
        .png, .bmp, .gif, .tiff, .tif (case-insensitive).

    Notes
    -----
    Extensions are matched case-insensitively. Returns empty list if no
    valid image files are found.
    """
    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".tiff",
        ".tif"
    }

    image_files = [
        f
        for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    return sorted(image_files)


# ========================
# CLI INTERFACE (for terminal use)
# ========================


def predict_single_image_cli(image_path):
    """
    CLI wrapper for single image prediction.

    Command-line interface function that loads model, predicts on a single
    image, and prints formatted results to stdout.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file to predict.

    Returns
    -------
    None

    Side Effects
    -----------
    - Prints device information to stdout.
    - Prints prediction result with confidence and probabilities to stdout.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Loading model...")

    result = predict_image(image_path, track_carbon=False)

    print(
        f"\nPrediction: {result['predicted_class']} \
        (class {result['predicted_idx']})"
    )
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nAll probabilities:")
    for class_name, prob in result["probabilities"].items():
        print(f"  {class_name}: {prob:.2%}")


def predict_folder_cli(folder_path):
    """
    CLI wrapper for batch prediction on folder of images.

    Command-line interface function that processes all valid images in a
    directory and prints formatted results to stdout.

    Parameters
    ----------
    folder_path : str or Path
        Path to directory containing images to predict.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If folder path does not exist or is not a directory, or if no valid
        image files are found.

    Side Effects
    -----------
    - Prints device information, progress updates, and summary to stdout.
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    def progress_callback(current, total, message):
        print(f"[{current}/{total}] {message}")

    batch_result = predict_batch(
        image_files, track_carbon=False, progress_callback=progress_callback
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    for result in batch_result["results"]:
        if result["status"] == "success":
            print(
                f"{result['filename']:<40} \
                    -> {result['predicted_class']} \
                        ({result['confidence']:.2%})"
            )
        else:
            print(f"{result['filename']:<40} -> ERROR: {result['error']}")
    print("=" * 60)


def main():
    """
    Main entry point for the prediction script when run from command line.

    Parses command-line arguments to determine whether to predict on a single
    image or batch process a folder. Falls back to cfg.SAMPLE_IMG_PATH if no
    arguments provided.

    Command-line Usage
    ------------------
    $ python predict.py                    # Uses default sample image
    $ python predict.py image.jpg          # Single image prediction
    $ python predict.py /path/to/folder/   # Batch folder prediction

    Parameters
    ----------
    None (reads from sys.argv)

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If invalid arguments or path does not exist.
    """
    if len(sys.argv) > 2:
        print("Usage: uv run predict.py <path_to_image_or_folder>")
        print("Examples:")
        print("  uv run predict.py img.jpg")
        print("  uv run predict.py /path/to/images/")
        sys.exit(1)
    elif len(sys.argv) == 1:
        image_path = cfg.SAMPLE_IMG_PATH
        predict_single_image_cli(image_path)
    else:
        input_path = Path(sys.argv[1])

        if not input_path.exists():
            print(f"Error: Path '{input_path}' does not exist.")
            sys.exit(1)

        if input_path.is_dir():
            predict_folder_cli(input_path)
        else:
            predict_single_image_cli(input_path)


if __name__ == "__main__":
    main()
