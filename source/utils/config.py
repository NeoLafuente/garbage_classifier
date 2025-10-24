#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module for Garbage Classification Project.

This module contains all configuration parameters and constants used
throughout the garbage classification project, including dataset paths,
model parameters, and class definitions.

Attributes
----------
PROJECT_ROOT : Path
    Root directory of the project (garbage_classifier/).
DATA_DIR : Path
    Directory containing all data (raw, processed, interim).
MODELS_DIR : Path
    Directory containing model weights and performance metrics.
DATASET_PATH : Path
    Path to the raw garbage dataset images.
LOSS_CURVES_PATH : Path
    Directory where training/validation curves are saved.
MODEL_PATH : Path
    Path to the best trained model checkpoint.
SAMPLE_IMG_PATH : Path
    Path to a sample image for testing predictions.
MODEL_URL : str
    GitHub Releases URL for downloading pretrained model.
CLASS_NAMES : list of str
    List of garbage category names for classification.
NUM_CLASSES : int
    Number of classification categories.
MAX_EPOCHS : int
    Maximum number of training epochs.
BATCH_SIZE : int
    Batch size for training and validation.
LEARNING_RATE : float
    Initial learning rate for optimizer.
NUM_WORKERS : int
    Number of worker processes for data loading.
IMAGE_SIZE : tuple
    Target image size for model input (height, width).
MEAN : list of float
    ImageNet normalization mean values for RGB channels.
STD : list of float
    ImageNet normalization standard deviation for RGB channels.

Examples
--------
>>> from source.utils import config as cfg
>>> model = GarbageClassifier(num_classes=cfg.NUM_CLASSES)
>>> trainer = pl.Trainer(max_epochs=cfg.MAX_EPOCHS)
>>> model_path = cfg.ensure_model_downloaded()

Notes
-----
All paths use pathlib.Path for cross-platform compatibility.
The pretrained model is automatically downloaded from GitHub Releases
if not found locally.
"""
__docformat__ = "numpy"

import os
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm
from platformdirs import user_data_dir

# ============================================
# APP NAME
# ============================================
APP_NAME = "garbage_classifier"

# ============================================
# Project Structure
# ============================================
DATA_DIR = "data"
MODELS_DIR = "models"

# ============================================
# Data Paths
# ============================================
RAW_DATA_DIR = f"{DATA_DIR}/raw"
DATASET_PATH = f"{RAW_DATA_DIR}/Garbage_Dataset_Classification/images"
SAMPLE_IMG_PATH = f"{RAW_DATA_DIR}/sample.jpg"
SAMPLE_DATASET_PATH = f"{RAW_DATA_DIR}/sample_dataset"
# ============================================
# Gradio Paths
# ============================================
APP_DIR = "app"
SECTIONS_DIR = f"{APP_DIR}/sections"
CACHED_DATA_DIR = f"{SECTIONS_DIR}/cached_data"

# ============================================
# Model Paths
# ============================================
WEIGHTS_DIR = f"{MODELS_DIR}/weights"
BEST_MODEL_DIR = f"{MODELS_DIR}/best"
PERFORMANCE_DIR = f"{MODELS_DIR}/performance"
LOSS_CURVES_PATH = f"{PERFORMANCE_DIR}/loss_curves"

MODEL_FILENAME = "model_resnet18_garbage.ckpt"
MODEL_PATH = f"{WEIGHTS_DIR}/{MODEL_FILENAME}"
EMISSIONS_FILE = "emissions.csv"

# ============================================
# Model Download Configuration
# ============================================
GITHUB_USER = "NeoLafuente"
GITHUB_REPO = "garbage_classifier"
MODEL_VERSION = "v0.1.0"

MODEL_URL = (
    f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/"
    f"releases/download/{MODEL_VERSION}/{MODEL_FILENAME}"
)

MODEL_PERFORMANCE_URL = (
    f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/"
    "releases/download/v0.1.4/performance.zip"
)

# ============================================
# Class Configuration
# ============================================
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASS_NAMES)

# ============================================
# Training Hyperparameters
# ============================================
MAX_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
PATIENCE = 10

# ============================================
# Model Configuration
# ============================================
IMAGE_SIZE = (224, 224)
MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
STD = [0.229, 0.224, 0.225]


# ============================================
# Utility Functions
# ============================================
def download_file(url: str, destination: Path, chunk_size: int = 8192) -> None:
    """
    Download a file from URL with progress bar.

    Parameters
    ----------
    url : str
        URL of the file to download.
    destination : Path
        Local path where to save the file.
    chunk_size : int, optional
        Download chunk size in bytes, by default 8192.

    Raises
    ------
    requests.HTTPError
        If download fails (e.g., 404, 403).
    requests.RequestException
        If network error occurs.

    Examples
    --------
    >>> download_file(
    ...     "https://example.com/model.ckpt",
    ...     Path("models/model.ckpt")
    ... )
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    print("📥 Downloading from GitHub Releases...")
    print(f"   URL: {url}")
    print(f"   Destination: {destination}")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = f.write(chunk)
            progress_bar.update(size)

    print(f"✅ Download completed: {destination}")


def ensure_model_downloaded() -> Path:
    """
    Ensure pretrained model is available, download if missing.

    Returns
    -------
    Path
        Path to the model checkpoint file.

    Raises
    ------
    requests.RequestException
        If model download fails.
    FileNotFoundError
        If model cannot be found or downloaded.

    Notes
    -----
    Downloads the pretrained model from GitHub Releases if not found locally.
    The model is saved in `models/best/model_resnet18_garbage.ckpt`.

    The download URL is constructed from:
    - GITHUB_USER: Your GitHub username
    - GITHUB_REPO: Repository name
    - MODEL_VERSION: Release tag (e.g., v0.1.0)
    - MODEL_FILENAME: Model checkpoint filename

    Examples
    --------
    >>> model_path = ensure_model_downloaded()
    >>> model = GarbageClassifier.load_from_checkpoint(model_path)
    """
    model_path = Path(f"{get_valid_dir(BEST_MODEL_DIR)}/{MODEL_FILENAME}")
    performance_path = model_path.parent / "performance.zip"

    if not model_path.exists():
        print("🔍 Pretrained model not found locally.")
        print(f"   Expected location: {model_path}")

        try:
            download_file(MODEL_URL, model_path)
            download_file(MODEL_PERFORMANCE_URL, performance_path)
            with zipfile.ZipFile(performance_path, "r") as zip_ref:
                zip_ref.extractall(performance_path.parent)
            os.remove(performance_path)
        except requests.HTTPError as e:
            print(f"\n❌ Failed to download model: {e}")
            raise FileNotFoundError(f"Model not found: {model_path}") from e
        except requests.RequestException as e:
            print(f"\n❌ Network error: {e}")
            print("\n💡 Check your internet connection or download manually:")
            print(f"   URL: {MODEL_URL}")
            print(f"   Destination: {model_path}")
            raise
    else:
        print(f"✅ Model found: {model_path}")

    emission_file_path = model_path.parent / "emissions.csv"
    if not emission_file_path.exists():
        with open(emission_file_path, 'w') as file:
            header = (
                "timestamp,project_name,run_id,experiment_id,duration,"
                "emissions,emissions_rate,cpu_power,gpu_power,ram_power,"
                "cpu_energy,gpu_energy,ram_energy,energy_consumed,"
                "country_name,country_iso_code,region,cloud_provider,"
                "cloud_region,os,python_version,codecarbon_version,"
                "cpu_count,cpu_model,gpu_count,gpu_model,longitude,latitude,"
                "ram_total_size,tracking_mode,on_cloud,pue"
            )
            file.write(header + "\n")

    return model_path


def get_valid_dir(local_path: str) -> Path:
    """
    Devuelve la carpeta local si existe.
    Si no existe, usa la de platformdirs y la crea si hace falta.
    """
    local_dir = Path(local_path)

    if local_dir.exists() and local_dir.is_dir():
        return str(local_dir.resolve())

    fallback_dir = Path(user_data_dir(APP_NAME), local_path)
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return str(fallback_dir.resolve())


def get_emissions_path():
    """
    Get the path to the emissions CSV file.

    Returns
    -------
    pathlib.Path
        Path object pointing to the emissions.csv file in the model directory.

    Notes
    -----
    The emissions file is stored in the same directory as the trained model
    checkpoint, as defined in the global configuration.
    """
    # return Path(cfg.MODEL_PATH).parent / "emissions.csv"
    return Path(f"{get_valid_dir(BEST_MODEL_DIR)}/{EMISSIONS_FILE}")
