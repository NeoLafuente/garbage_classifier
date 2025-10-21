# source/predict.py
#!/usr/bin/env python3
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


# ========================
# CORE PREDICTION FUNCTIONS (importable)
# ========================

def load_model_for_inference(model_path=None, device=None):
    """
    Load a trained model for inference.
    
    Parameters
    ----------
    model_path : str or Path, optional
        Path to model checkpoint. If None, uses cfg.MODEL_PATH
    device : torch.device, optional
        Device to load model on. If None, auto-selects GPU/CPU
        
    Returns
    -------
    tuple of (GarbageClassifier, torch.device, torchvision.transforms.Compose)
        Loaded model, device, and image transform pipeline
        
    Examples
    --------
    >>> model, device, transform = load_model_for_inference()
    >>> # Use in Gradio or other applications
    """
    if model_path is None:
        model_path = cfg.MODEL_PATH
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GarbageClassifier.load_from_checkpoint(
        model_path,
        num_classes=cfg.NUM_CLASSES
    )
    model = model.to(device)
    model.eval()
    
    transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    
    return model, device, transform


def predict_image(image_path, model=None, transform=None, device=None, 
                  class_names=None, track_carbon=False):
    """
    Predict the garbage category of an input image.

    Parameters
    ----------
    image_path : str, Path, or PIL.Image
        Path to image file or PIL Image object
    model : GarbageClassifier, optional
        Loaded model. If None, loads from cfg.MODEL_PATH
    transform : torchvision.transforms.Compose, optional
        Image transformation pipeline. If None, uses default ResNet18 transforms
    device : torch.device, optional
        Device for inference. If None, auto-selects GPU/CPU
    class_names : list of str, optional
        Class names. If None, uses cfg.CLASS_NAMES
    track_carbon : bool, default=False
        Whether to track carbon emissions

    Returns
    -------
    dict
        Dictionary containing:
        - 'predicted_class': str, predicted class name
        - 'predicted_idx': int, predicted class index
        - 'confidence': float, confidence score (0-1)
        - 'probabilities': dict, all class probabilities
        - 'emissions': dict or None, carbon emissions data if tracked

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
            output_dir=str(Path(cfg.MODEL_PATH).parent),
            log_level="warning"
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
            class_names[i]: probs[i].item() 
            for i in range(len(class_names))
        }
        
        # Stop carbon tracking
        if track_carbon:
            emissions_kg = tracker.stop()
            from source.utils.carbon_utils import kg_co2_to_car_distance, format_car_distance
            car_distances = kg_co2_to_car_distance(emissions_kg)
            emissions_data = {
                'emissions_kg': emissions_kg,
                'emissions_g': emissions_kg * 1000,
                'car_distance_km': car_distances['distance_km'],
                'car_distance_m': car_distances['distance_m'],
                'car_distance_formatted': format_car_distance(emissions_kg)
            }
        
        return {
            'predicted_class': pred_class,
            'predicted_idx': pred_idx,
            'confidence': confidence,
            'probabilities': all_probs,
            'emissions': emissions_data,
            'image': image  # Return PIL image for display
        }
    
    except Exception as e:
        if track_carbon:
            tracker.stop()
        raise e


def predict_batch(image_paths, model=None, transform=None, device=None,
                  class_names=None, track_carbon=False, progress_callback=None):
    """
    Predict garbage categories for multiple images.
    
    Parameters
    ----------
    image_paths : list of (str or Path)
        List of image file paths
    model : GarbageClassifier, optional
        Loaded model. If None, loads from cfg.MODEL_PATH
    transform : torchvision.transforms.Compose, optional
        Image transformation pipeline
    device : torch.device, optional
        Device for inference
    class_names : list of str, optional
        Class names
    track_carbon : bool, default=False
        Whether to track carbon emissions
    progress_callback : callable, optional
        Callback function(current, total, message) for progress updates
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'results': list of prediction dicts
        - 'summary': summary statistics
        - 'emissions': carbon emissions data if tracked
        
    Examples
    --------
    >>> results = predict_batch(["img1.jpg", "img2.jpg"], track_carbon=True)
    >>> for r in results['results']:
    ...     print(f"{r['filename']}: {r['predicted_class']}")
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
            output_dir=str(Path(cfg.MODEL_PATH).parent),
            log_level="warning"
        )
        tracker.start()
    
    results = []
    total = len(image_paths)
    
    for idx, image_path in enumerate(image_paths):
        if progress_callback:
            progress_callback(idx + 1, total, f"Processing {Path(image_path).name}")
        
        try:
            result = predict_image(
                image_path, 
                model=model, 
                transform=transform, 
                device=device,
                class_names=class_names,
                track_carbon=False  # Don't track individual images
            )
            result['filename'] = Path(image_path).name
            result['status'] = 'success'
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': Path(image_path).name,
                'status': 'error',
                'error': str(e)
            })
    
    # Stop carbon tracking
    if track_carbon:
        emissions_kg = tracker.stop()
        from source.utils.carbon_utils import kg_co2_to_car_distance, format_car_distance
        car_distances = kg_co2_to_car_distance(emissions_kg)
        emissions_data = {
            'emissions_kg': emissions_kg,
            'emissions_g': emissions_kg * 1000,
            'car_distance_km': car_distances['distance_km'],
            'car_distance_m': car_distances['distance_m'],
            'car_distance_formatted': format_car_distance(emissions_kg),
            'emissions_per_image_g': (emissions_kg * 1000) / len(image_paths)
        }
    
    # Summary
    successful = len([r for r in results if r.get('status') == 'success'])
    summary = {
        'total_images': total,
        'successful': successful,
        'failed': total - successful
    }
    
    return {
        'results': results,
        'summary': summary,
        'emissions': emissions_data
    }


def get_image_files(path):
    """
    Get all valid image files from a directory.
    
    [Keep existing implementation - no changes needed]
    """
    valid_extensions = {
        '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif'
    }
    image_files = [
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    return sorted(image_files)


# ========================
# CLI INTERFACE (for terminal use)
# ========================

def predict_single_image_cli(image_path):
    """CLI wrapper for single image prediction"""
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("Loading model...")
    
    result = predict_image(image_path, track_carbon=False)
    
    print(f"\nPrediction: {result['predicted_class']} (class {result['predicted_idx']})")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nAll probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"  {class_name}: {prob:.2%}")


def predict_folder_cli(folder_path):
    """CLI wrapper for folder prediction"""
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
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    def progress_callback(current, total, message):
        print(f"[{current}/{total}] {message}")
    
    batch_result = predict_batch(
        image_files, 
        track_carbon=False,
        progress_callback=progress_callback
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    for result in batch_result['results']:
        if result['status'] == 'success':
            print(f"{result['filename']:<40} -> {result['predicted_class']} ({result['confidence']:.2%})")
        else:
            print(f"{result['filename']:<40} -> ERROR: {result['error']}")
    print("=" * 60)


def main():
    """
    Main entry point for the prediction script.
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