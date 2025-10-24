#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluation and Inference Interface for Gradio Application.

This module provides comprehensive model evaluation tools and inference
capabilities through a Gradio interface. It includes visualization of
model performance metrics, confusion matrices, calibration curves, and
both single-image and batch prediction functionality.

Key features include:
- Multi-model support (best model and latest trained model)
- Cached computation of expensive metrics (confusion matrices, calibration)
- Automatic cache invalidation based on model modification time
- Real-time inference with carbon emissions tracking
- Training metrics visualization (loss/accuracy curves)

Notes
-----
Cache files are stored in `app/sections/cached_data/` and are automatically
invalidated when the corresponding model file is updated. This ensures
that evaluation metrics reflect the current model state while avoiding
redundant computation.
"""
__docformat__ = "numpy"

import gradio as gr
from source.utils.carbon_utils import format_total_emissions_display
from source.utils import config as cfg
from source.utils.custom_classes.EvalAnalyzer import GarbageModelAnalyzer
from source.predict import (
    predict_image,
    predict_batch,
    load_model_for_inference
)
from source.utils.config import get_valid_dir as gvd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import pickle


def get_available_models():
    """
    Get dictionary of available trained models.

    Returns
    -------
    dict of {str: str}
        Dictionary mapping model display names to their checkpoint file paths.
        Includes both the best provided model and the latest trained model.

    Examples
    --------
    >>> models = get_available_models()
    >>> models
    {
        'Best Model (Provided)': 'models/best/model_resnet18_garbage.ckpt',
        'Latest Trained Model': 'models/resnet18_garbage.ckpt'
    }
    """
    best_model_path = cfg.ensure_model_downloaded()
    models_dict = {
        "Best Model (Provided)": str(best_model_path),
        "Latest Trained Model": str(
            f"{gvd(cfg.WEIGHTS_DIR)}/model_resnet18_garbage.ckpt"
        ),
    }
    return models_dict


# Setup cache directory
CACHE_DIR = Path("app/sections/cached_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def is_cache_valid(cache_file, model_path):
    """
    Check if cached file is newer than the model checkpoint.

    Validates cache by comparing modification timestamps. Cache is considered
    valid only if it was created/modified after the model checkpoint.

    Parameters
    ----------
    cache_file : pathlib.Path
        Path to the cached file to validate.
    model_path : str or pathlib.Path
        Path to the model checkpoint file.

    Returns
    -------
    bool
        True if cache exists and is newer than the model, False otherwise.

    Notes
    -----
    This function implements automatic cache invalidation to ensure that
    evaluation metrics are regenerated when a model is retrained.

    Examples
    --------
    >>> cache_file = Path("cached_data/cm_raw_Latest.pkl")
    >>> model_path = "models/resnet18_garbage.ckpt"
    >>> is_cache_valid(cache_file, model_path)
    False  # Cache doesn't exist or is older than model
    """
    if not Path(cache_file).exists():
        return False

    # TODO: gvd(model_path.parent)
    if model_path is None or not Path(model_path).exists():
        return False

    cache_time = Path(cache_file).stat().st_mtime
    model_time = Path(model_path).stat().st_mtime

    print(cache_file, model_path)
    print(cache_time, model_time)

    return cache_time > model_time  # Cache is valid if newer than model


# State to hold both confusion matrices
confusion_matrices_state = {
    "raw": None,
    "normalized": None,
    "model_choice": None,  # Track which model generated these
}


# ========================
# METRICS FUNCTIONS
# ========================


def generate_confusion_matrix(
    model_choice,
    show_normalized,
    progress=gr.Progress()
):
    """
    Generate and cache both raw and normalized confusion matrices.

    This function generates confusion matrices for the validation set,
    caching both raw and normalized versions to enable instant toggling
    without recomputation. Implements three-tier caching: memory, disk,
    and automatic invalidation.

    Parameters
    ----------
    model_choice : str
        Name of the selected model (from get_available_models()).
    show_normalized : bool
        Whether to display the normalized version initially.
    progress : gr.Progress, optional
        Gradio progress tracker for UI updates.

    Returns
    -------
    tuple of (matplotlib.figure.Figure or None, str, gr.update)
        - Figure: The confusion matrix plot (raw or normalized)
        - str: Status message describing the operation result
        - gr.update: Gradio update object to control checkbox visibility

    Notes
    -----
    Caching strategy:
    1. Check memory cache (fastest)
    2. Check disk cache with validation (fast)
    3. Regenerate if cache invalid or missing (slow)

    Both raw and normalized matrices are always generated together and
    stored separately to enable instant toggling via the checkbox.

    The cache is automatically invalidated when the model file is modified,
    ensuring metrics reflect the current model state.

    See Also
    --------
    toggle_confusion_matrix : Switch between raw/normalized without
    regenerating.
    is_cache_valid : Cache validation logic
    """
    global confusion_matrices_state # TODO: Revert this (uncomment)

    if not model_choice:
        return None, "Please select a model first", gr.update(visible=False)

    # Get model path
    models_dict = get_available_models()
    model_path = models_dict.get(model_choice)

    # Check if we already have matrices for this model in memory
    if (
        confusion_matrices_state["model_choice"] == model_choice
        and confusion_matrices_state["raw"] is not None
        and confusion_matrices_state["normalized"] is not None
    ):

        selected_matrix = (
            confusion_matrices_state["normalized"]
            if show_normalized
            else confusion_matrices_state["raw"]
        )
        matrix_type = "Normalized" if show_normalized else "Raw"
        return (
            selected_matrix,
            f"✅ {matrix_type} confusion matrix (from memory)",
            gr.update(visible=True, interactive=True),
        )

    # Check disk cache
    cache_file_raw = \
        CACHE_DIR / f"cm_raw_{model_choice.replace(' ', '_')}.pkl"
    cache_file_norm = \
        CACHE_DIR / f"cm_norm_{model_choice.replace(' ', '_')}.pkl"

    if is_cache_valid(cache_file_raw, model_path) and is_cache_valid(
        cache_file_norm, model_path
    ):
        try:
            progress(0.2, desc="Loading from cache...")
            with open(cache_file_raw, "rb") as f:
                fig_raw = pickle.load(f)
            with open(cache_file_norm, "rb") as f:
                fig_norm = pickle.load(f)

            confusion_matrices_state["raw"] = fig_raw
            confusion_matrices_state["normalized"] = fig_norm
            confusion_matrices_state["model_choice"] = model_choice

            selected_matrix = fig_norm if show_normalized else fig_raw
            matrix_type = "Normalized" if show_normalized else "Raw"
            progress(1.0, desc="Done!")
            return (
                selected_matrix,
                f"✅ {matrix_type} confusion matrix loaded from cache",
                gr.update(visible=True, interactive=True),
            )
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}")

    # Generate new matrices
    try:
        progress(0.1, desc="Loading model...")
        analyzer = GarbageModelAnalyzer()
        analyzer.load_model(checkpoint_path=model_path)

        progress(0.3, desc="Setting up data...")
        analyzer.setup_data(batch_size=32)

        progress(0.5, desc="Evaluating model...")
        val_loader = analyzer.data_module.val_dataloader()
        preds, labels, probs = analyzer.evaluate_loader(val_loader)

        progress(0.7, desc="Generating confusion matrices...")

        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        num_classes = cfg.NUM_CLASSES
        cm_raw = confusion_matrix(
            labels.cpu().numpy(),
            preds.cpu().numpy(),
            labels=range(num_classes)
        )
        cm_norm = cm_raw.astype("float") / cm_raw.sum(axis=1)[:, np.newaxis]

        # Generate RAW matrix figure
        fig_raw, ax_raw = plt.subplots(figsize=(10, 8))
        disp_raw = ConfusionMatrixDisplay(
            confusion_matrix=cm_raw, display_labels=cfg.CLASS_NAMES
        )
        disp_raw.plot(cmap=plt.cm.Blues, ax=ax_raw)
        ax_raw.set_title("Confusion Matrix - Validation Set")
        plt.tight_layout()

        # Generate NORMALIZED matrix figure
        fig_norm, ax_norm = plt.subplots(figsize=(10, 8))
        disp_norm = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm, display_labels=cfg.CLASS_NAMES
        )
        disp_norm.plot(cmap=plt.cm.Blues, ax=ax_norm)
        ax_norm.set_title("Normalized Confusion Matrix - Validation Set")
        plt.tight_layout()

        # Save to cache
        progress(0.9, desc="Saving to cache...")
        with open(cache_file_raw, "wb") as f:
            pickle.dump(fig_raw, f)
        with open(cache_file_norm, "wb") as f:
            pickle.dump(fig_norm, f)

        # Update state
        confusion_matrices_state["raw"] = fig_raw
        confusion_matrices_state["normalized"] = fig_norm
        confusion_matrices_state["model_choice"] = model_choice

        selected_matrix = fig_norm if show_normalized else fig_raw
        matrix_type = "Normalized" if show_normalized else "Raw"

        progress(1.0, desc="Done!")
        return (
            selected_matrix,
            f"✅ {matrix_type} confusion matrix generated and cached",
            gr.update(visible=True, interactive=True),
        )

    except Exception as e:
        return None, f"❌ Error: {str(e)}", gr.update(visible=False)


def toggle_confusion_matrix(show_normalized):
    """
    Toggle between raw and normalized confusion matrices instantly.

    Switches the displayed confusion matrix without regenerating, using
    cached versions from memory. This enables instant UI response to the
    normalization checkbox.

    Parameters
    ----------
    show_normalized : bool
        If True, return normalized matrix; if False, return raw matrix.

    Returns
    -------
    matplotlib.figure.Figure or None
        The requested confusion matrix figure, or None if not available
        in memory cache.

    Notes
    -----
    This function only accesses the in-memory cache. If matrices are not
    yet generated, it returns None. The generate_confusion_matrix function
    should be called first to populate the cache.

    See Also
    --------
    generate_confusion_matrix : Generate and cache both matrix versions
    """
    global confusion_matrices_state # TODO: Revert this (uncomment)

    if show_normalized:
        if confusion_matrices_state["normalized"] is not None:
            return confusion_matrices_state["normalized"]
    else:
        if confusion_matrices_state["raw"] is not None:
            return confusion_matrices_state["raw"]

    return None


def generate_calibration_curves(model_choice, progress=gr.Progress()):
    """
    Generate and cache calibration curves for the selected model.

    Calibration curves show how well predicted probabilities match actual
    outcomes. This function generates curves for all classes and caches
    the result for faster subsequent access.

    Parameters
    ----------
    model_choice : str
        Name of the selected model (from get_available_models()).
    progress : gr.Progress, optional
        Gradio progress tracker for UI updates.

    Returns
    -------
    tuple of (matplotlib.figure.Figure or None, str)
        - Figure: The calibration curves plot
        - str: Status message describing the operation result

    Notes
    -----
    The calibration plot is generated using GarbageModelAnalyzer's
    plot_calibration_curves method and cached to disk. Cache is
    automatically invalidated when the model is retrained.

    Well-calibrated models should have curves close to the diagonal,
    indicating that predicted probabilities match actual frequencies.

    See Also
    --------
    is_cache_valid : Cache validation logic
    GarbageModelAnalyzer.plot_calibration_curves : Core plotting function
    """
    if not model_choice:
        return None, "Please select a model first"

    # Get model path
    models_dict = get_available_models()
    model_path = models_dict.get(model_choice)

    # Check disk cache
    cache_file = CACHE_DIR / f"calib_{model_choice.replace(' ', '_')}.pkl"

    if is_cache_valid(cache_file, model_path):
        try:
            progress(0.2, desc="Loading from cache...")
            with open(cache_file, "rb") as f:
                fig = pickle.load(f)
            progress(1.0, desc="Done!")
            return fig, "✅ Calibration curves loaded from cache"
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}")

    try:
        progress(0.1, desc="Loading model...")
        analyzer = GarbageModelAnalyzer()
        analyzer.load_model(checkpoint_path=model_path)

        progress(0.3, desc="Setting up data...")
        analyzer.setup_data(batch_size=32)

        progress(0.5, desc="Evaluating model...")
        val_loader = analyzer.data_module.val_dataloader()
        preds, labels, probs = analyzer.evaluate_loader(val_loader)

        progress(0.8, desc="Generating calibration curves...")

        analyzer.plot_calibration_curves(labels, probs)
        fig = plt.gcf()

        # Save to cache
        progress(0.9, desc="Saving to cache...")
        with open(cache_file, "wb") as f:
            pickle.dump(fig, f)

        progress(1.0, desc="Done!")
        return fig, "✅ Calibration curves generated and cached"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def get_metrics_path_for_model(model_choice):
    """
    Get the correct metrics.json path for the selected model.

    Different models store their training metrics in different locations.
    This function maps model choices to their corresponding metrics files.

    Parameters
    ----------
    model_choice : str
        Name of the selected model (from get_available_models()).

    Returns
    -------
    pathlib.Path
        Path to the metrics.json file for the selected model.

    Notes
    -----
    The best provided model has metrics in a fixed location, while the
    latest trained model uses the configured loss curves path.
    """
    if model_choice == "Best Model (Provided)":
        return Path(gvd("models/best/performance/loss_curves")) / "metrics.json"
    else:
        return Path(gvd(cfg.LOSS_CURVES_PATH)) / "metrics.json"


def load_loss_curves(model_choice):
    """
    Load and plot training/validation loss curves from metrics.json.

    Parameters
    ----------
    model_choice : str
        Name of the selected model (from get_available_models()).

    Returns
    -------
    tuple of (matplotlib.figure.Figure or None, str)
        - Figure: Loss curves plot showing train and validation loss
        - str: Status message describing the operation result

    Notes
    -----
    Loss curves show how training and validation loss evolved during
    training. Diverging curves may indicate overfitting, while parallel
    curves suggest good generalization.

    The plot includes:
    - Blue line with circles: Training loss
    - Red line with squares: Validation loss
    - Grid for easier reading

    See Also
    --------
    load_accuracy_curves : Load accuracy metrics instead of loss
    get_metrics_path_for_model : Determine metrics file location
    """
    try:
        metrics_path = Path(get_metrics_path_for_model(model_choice))

        if not metrics_path.exists():
            return (
                None,
                f"❌ No training metrics found for \
                    {model_choice}. Path: {metrics_path}",
            )

        with open(metrics_path, "r") as f:
            data = json.load(f)

        train_losses = data.get("train_losses", [])
        val_losses = data.get("val_losses", [])

        if not train_losses and not val_losses:
            return None, "❌ No loss data available"

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        if train_losses:
            ax.plot(
                epochs,
                train_losses,
                "b-o",
                label="Train Loss",
                linewidth=2
            )
        if val_losses:
            ax.plot(
                epochs,
                val_losses,
                "r-s",
                label="Validation Loss",
                linewidth=2
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"Loss Curves - \
            {model_choice}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig, f"✅ Loss curves loaded successfully from {model_choice}"

    except Exception as e:
        return None, f"❌ Error loading loss curves: {str(e)}"


def load_accuracy_curves(model_choice):
    """
    Load and plot training/validation accuracy curves from metrics.json.

    Parameters
    ----------
    model_choice : str
        Name of the selected model (from get_available_models()).

    Returns
    -------
    tuple of (matplotlib.figure.Figure or None, str)
        - Figure: Accuracy curves plot showing train and validation accuracy
        - str: Status message describing the operation result

    Notes
    -----
    Accuracy curves show how classification accuracy evolved during training.
    The plot includes:
    - Blue line with circles: Training accuracy
    - Red line with squares: Validation accuracy
    - Y-axis limited to [0, 1] for consistency
    - Grid for easier reading

    A large gap between train and validation accuracy indicates overfitting.

    See Also
    --------
    load_loss_curves : Load loss metrics instead of accuracy
    get_metrics_path_for_model : Determine metrics file location
    """
    try:
        metrics_path = Path(get_metrics_path_for_model(model_choice))

        if not metrics_path.exists():
            return (
                None,
                f"❌ No training metrics found for \
                    {model_choice}. Path: {metrics_path}",
            )

        with open(metrics_path, "r") as f:
            data = json.load(f)

        train_accs = data.get("train_accs", [])
        val_accs = data.get("val_accs", [])

        if not train_accs and not val_accs:
            return None, "❌ No accuracy data available"

        fig, ax = plt.subplots(figsize=(10, 6))

        if train_accs:
            ax.plot(
                range(1, len(train_accs) + 1),
                train_accs,
                "b-o",
                label="Train Accuracy",
                linewidth=2,
            )
        if val_accs:
            ax.plot(
                range(1, len(val_accs) + 1),
                val_accs,
                "r-s",
                label="Validation Accuracy",
                linewidth=2,
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            f"Accuracy Curves - \
                {model_choice}", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        plt.tight_layout()

        return fig, f"✅ Accuracy curves \
            loaded successfully from {model_choice}"

    except Exception as e:
        return None, f"❌ Error loading \
            accuracy curves: {str(e)}"


# ========================
# PREDICTION FUNCTIONS
# ========================


def predict_single_image_gradio(
    model_choice, image, carbon_display_text, track_carbon=True
):
    """
    Gradio wrapper for single image prediction with visualization.

    Performs inference on a single image and generates a horizontal bar
    chart of class probabilities, highlighting the predicted class.

    Parameters
    ----------
    model_choice : str
        Name of the selected model (from get_available_models()).
    image : np.ndarray
        Input image as numpy array (from gr.Image component).
    carbon_display_text : str
        Current HTML string for the carbon display (to be updated).
    track_carbon : bool, optional
        Whether to track carbon emissions for this prediction,
        by default True.

    Returns
    -------
    tuple of (matplotlib.figure.Figure or None, str, str)
        - Figure: Bar chart of class probabilities
        - str: Markdown-formatted prediction results and statistics
        - str: Updated HTML for carbon display

    Notes
    -----
    The probability bar chart uses:
    - Green bar for the predicted class
    - Sky blue bars for other classes
    - Percentage labels on each bar

    If carbon tracking is enabled, emissions are added to the cumulative
    total and the carbon display is updated.

    See Also
    --------
    predict_batch : Batch prediction on multiple images
    source.predict.predict_image : Core prediction function
    """
    if image is None:
        return None, "Please upload an image", carbon_display_text

    if not model_choice:
        return None, "Please select a model first", carbon_display_text

    try:
        models_dict = get_available_models()
        model_path = models_dict.get(model_choice)
        model, device, transform = load_model_for_inference(
            model_path=model_path
        )

        result = predict_image(
            image,
            model=model,
            transform=transform,
            device=device,
            track_carbon=track_carbon,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        probs_list = [result["probabilities"][cls] for cls in cfg.CLASS_NAMES]
        pred_idx = result["predicted_idx"]
        aux_list = range(len(cfg.CLASS_NAMES))
        colors = [
            "green" if i == pred_idx else "skyblue" for i in aux_list
        ]
        bars = ax.barh(cfg.CLASS_NAMES, probs_list, color=colors)
        ax.set_xlabel("Probability", fontsize=12)
        ax.set_title(
            f'Prediction Probabilities\nPredicted Class: \
                {result["predicted_class"]}',
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlim([0, 1])

        for i, (bar, prob) in enumerate(zip(bars, probs_list)):
            ax.text(
                prob + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()

        result_text = f"### 🎯 Prediction: \
        **{result['predicted_class']}**\n\n"
        result_text += f"**Confidence:** {result['confidence']*100:.2f}%\n\n"
        result_text += "**All Probabilities:**\n"
        for class_name, prob in result["probabilities"].items():
            emoji = "🏆" if class_name == result["predicted_class"] \
                else "  "
            result_text += f"{emoji} {class_name}: {prob*100:.2f}%\n"

        updated_carbon_display = carbon_display_text
        if result["emissions"]:
            emissions = result["emissions"]
            result_text += "\n\n### 🌍 Carbon Footprint\n"
            result_text += f"- **Emissions:** \
                {emissions['emissions_g']:.4f}g CO₂eq\n"
            result_text += f"- **🚗 Car equivalent:** \
                {emissions['car_distance_formatted']} driven\n"
            updated_carbon_display = format_total_emissions_display(
                cfg.get_emissions_path()
            )

        return fig, result_text, updated_carbon_display

    except Exception as e:
        return None, f"❌ Error: {str(e)}", carbon_display_text


def predict_folder_gradio(
    model_choice, files, carbon_display_text, track_carbon=True
):
    """
    Gradio wrapper for batch prediction on multiple images.

    Performs inference on multiple images simultaneously and returns
    results in a tabular format with per-image predictions and overall
    statistics.

    Parameters
    ----------
    model_choice : str
        Name of the selected model (from get_available_models()).
    files : list of gr.File
        List of uploaded file objects from Gradio file input.
    carbon_display_text : str
        Current HTML string for the carbon display (to be updated).
    track_carbon : bool, optional
        Whether to track carbon emissions for this batch, by default True.

    Returns
    -------
    tuple of (pd.DataFrame or None, str, str)
        - DataFrame: Table with filename, predicted class, and confidence
          for each image
        - str: Markdown-formatted summary statistics and carbon footprint
        - str: Updated HTML for carbon display

    Notes
    -----
    The results table includes:
    - Filename: Original image filename
    - Predicted Class: Predicted garbage category
    - Confidence (%): Prediction confidence as percentage

    Summary statistics include:
    - Total images processed
    - Number of successful predictions
    - Total carbon emissions (if tracked)
    - Average emissions per image
    - Car distance equivalent

    Failed predictions are marked as "Error" in the table with the
    error message in the confidence column.

    See Also
    --------
    predict_single_image_gradio : Single image prediction
    source.predict.predict_batch : Core batch prediction function
    """
    if not files or len(files) == 0:
        return None, "Please upload images", carbon_display_text

    if not model_choice:
        return None, "Please select a model first", carbon_display_text

    try:
        models_dict = get_available_models()
        model_path = models_dict.get(model_choice)
        model, device, transform = load_model_for_inference(
            model_path=model_path
        )

        image_paths = [file.name for file in files]

        batch_result = predict_batch(
            image_paths,
            model=model,
            transform=transform,
            device=device,
            track_carbon=track_carbon,
        )

        df_data = []
        for result in batch_result["results"]:
            if result["status"] == "success":
                df_data.append(
                    {
                        "Filename": result["filename"],
                        "Predicted Class": result["predicted_class"],
                        "Confidence (%)": f"{result['confidence']*100:.2f}",
                    }
                )
            else:
                df_data.append(
                    {
                        "Filename": result["filename"],
                        "Predicted Class": "Error",
                        "Confidence (%)": result["error"],
                    }
                )

        df_results = pd.DataFrame(df_data)

        summary = batch_result["summary"]
        result_text = "### 📊 Batch Prediction Results\n\n"
        result_text += f"**Total images processed:** \
            {summary['total_images']}\n"
        result_text += f"**Successful predictions:** \
            {summary['successful']}\n\n"

        updated_carbon_display = carbon_display_text
        if batch_result["emissions"]:
            emissions = batch_result["emissions"]
            result_text += "### 🌍 Carbon Footprint\n"
            result_text += f"- **Emissions:** \
                {emissions['emissions_g']:.4f}g CO₂eq\n"
            result_text += f"- **🚗 Car equivalent:** \
                {emissions['car_distance_formatted']} driven\n"
            result_text += f"- **Avg per image:** \
                {emissions['emissions_per_image_g']:.4f}g CO₂eq\n"
            updated_carbon_display = format_total_emissions_display(
                cfg.get_emissions_path()
            )

        return df_results, result_text, updated_carbon_display

    except Exception as e:
        return None, f"❌ Error: {str(e)}", carbon_display_text


# ========================
# UI LAYOUT
# ========================


def model_evaluation_tab(carbon_display):
    """
    Create the Model Evaluation and Inference UI section.

    Builds a comprehensive Gradio interface for model evaluation, metrics
    visualization, and real-time inference. Organized into three main areas:
    model selection, metrics visualization, and image prediction.

    Parameters
    ----------
    carbon_display : gr.HTML
        The carbon counter display component to update after inference
        operations. Shows cumulative emissions across all tracked operations.

    Returns
    -------
    list
        Empty list (kept for API consistency).

    Notes
    -----
    The interface is organized into sections:

    1. **Model Selection:**
       - Radio buttons to choose between best model and latest trained model

    2. **Metrics Visualization Tabs:**
       - Confusion Matrix: Raw and normalized versions with instant toggle
       - Loss Curves: Training and validation loss over epochs
       - Accuracy Curves: Training and validation accuracy over epochs
       - Calibration Curves: Per-class calibration analysis

    3. **Inference Tabs:**
       - Single Image: Upload and classify individual images
       - Batch Prediction: Process multiple images simultaneously

    All expensive computations (confusion matrices, calibration curves) are
    cached and automatically invalidated when models are retrained.

    Carbon emissions can be optionally tracked for all inference operations
    and are displayed in both absolute terms and car distance equivalents.

    Examples
    --------
    >>> with gr.Blocks() as demo:
    ...     carbon_display = gr.HTML()
    ...     model_evaluation_tab(carbon_display)
    """
    with gr.Column():
        gr.Markdown("### 🔬 Model Evaluation & Inference")
        gr.Markdown(
            "Evaluate trained models, visualize metrics, \
                and make predictions on new images."
        )

        gr.Markdown("#### 🧠 Model Selection")
        model_choice = gr.Radio(
            choices=list(get_available_models().keys()),
            value=list(get_available_models().keys())[0],
            label="Select Model",
            info="Choose between the best provided model or \
                your latest trained model",
        )

        gr.Markdown("---")

        gr.Markdown("#### 📈 Model Metrics & Visualizations")

        with gr.Tabs():
            with gr.Tab("Confusion Matrix"):
                show_normalized = gr.Checkbox(
                    label="Show Normalized",
                    value=False,
                    info="Toggle between raw counts and normalized \
                        percentages",
                    visible=False,
                )
                cm_button = gr.Button(
                    "Generate Confusion Matrix", variant="primary"
                )
                cm_plot = gr.Plot(label="Confusion Matrix")
                cm_status = gr.Markdown("")

                cm_button.click(
                    fn=generate_confusion_matrix,
                    inputs=[model_choice, show_normalized],
                    outputs=[cm_plot, cm_status, show_normalized],
                )

                show_normalized.change(
                    fn=toggle_confusion_matrix,
                    inputs=[show_normalized],
                    outputs=[cm_plot],
                )

            with gr.Tab("Loss Curves"):
                loss_button = gr.Button("Load Loss Curves", variant="primary")
                loss_plot = gr.Plot(label="Loss Curves")
                loss_status = gr.Markdown("")

                loss_button.click(
                    fn=load_loss_curves,
                    inputs=[model_choice],
                    outputs=[loss_plot, loss_status],
                )

            with gr.Tab("Accuracy Curves"):
                acc_button = gr.Button(
                    "Load Accuracy Curves", variant="primary"
                )
                acc_plot = gr.Plot(label="Accuracy Curves")
                acc_status = gr.Markdown("")

                acc_button.click(
                    fn=load_accuracy_curves,
                    inputs=[model_choice],
                    outputs=[acc_plot, acc_status],
                )

            with gr.Tab("Calibration Curves"):
                calib_button = gr.Button(
                    "Generate Calibration Curves", variant="primary"
                )
                calib_plot = gr.Plot(label="Calibration Curves")
                calib_status = gr.Markdown("")

                calib_button.click(
                    fn=generate_calibration_curves,
                    inputs=[model_choice],
                    outputs=[calib_plot, calib_status],
                )

        gr.Markdown("---")

        gr.Markdown("#### 🔍 Image Prediction")

        with gr.Tabs():
            with gr.Tab("Single Image"):
                gr.Markdown(
                    "Upload an image to classify it into one of \
                        the garbage categories."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        single_image_input = gr.Image(
                            label="Upload Image", type="numpy", height=400
                        )
                        single_track_carbon = gr.Checkbox(
                            label="🌍 Track Carbon Emissions", value=True
                        )
                        single_predict_button = gr.Button(
                            "🔍 Predict", variant="primary", size="lg"
                        )

                    with gr.Column(scale=1):
                        single_probs_plot = gr.Plot(
                            label="Class Probabilities"
                        )

                single_result_text = gr.Markdown(
                    "Upload an image and click 'Predict'"
                )

                single_predict_button.click(
                    fn=predict_single_image_gradio,
                    inputs=[
                        model_choice,
                        single_image_input,
                        carbon_display,
                        single_track_carbon,
                    ],
                    outputs=[
                        single_probs_plot,
                        single_result_text,
                        carbon_display
                    ],
                )

            with gr.Tab("Batch Prediction"):
                gr.Markdown("Upload multiple images \
                    to classify them all at once.")

                batch_image_input = gr.File(
                    label="Upload Images",
                    file_count="multiple", file_types=["image"]
                )
                batch_track_carbon = gr.Checkbox(
                    label="🌍 Track Carbon Emissions", value=True
                )
                batch_predict_button = gr.Button(
                    "🔍 Predict All", variant="primary", size="lg"
                )

                batch_result_text = gr.Markdown(
                    "Upload images and click 'Predict All'"
                )
                batch_results_table = gr.Dataframe(
                    label="Prediction Results",
                    interactive=False, wrap=True
                )

                batch_predict_button.click(
                    fn=predict_folder_gradio,
                    inputs=[
                        model_choice,
                        batch_image_input,
                        carbon_display,
                        batch_track_carbon,
                    ],
                    outputs=[
                        batch_results_table,
                        batch_result_text,
                        carbon_display
                    ],
                )

        gr.Markdown("---")
        gr.Markdown(
            "**ℹ️ Info:** Carbon emissions are tracked for \
                inference operations and added to the total \
                    carbon footprint."
        )

    return []
