# Project Structure and Usage Guide
## Project Organization
```
.
├── data
│   ├── interim
│   ├── processed
│   └── raw
│       └── Garbage_Dataset_Classification
│           ├── images
│           │   ├── cardboard
│           │   ├── glass
│           │   ├── metal
│           │   ├── paper
│           │   ├── plastic
│           │   └── trash
│           └── metadata.csv
├── docs
│   ├── index.html
│   ├── predict.html
│   ├── search.js
│   ├── train.html
│   └── utils
│       ├── config.html
│       └── custom_classes
│           ├── GarbageClassifier.html
│           ├── GarbageDataModule.html
│           └── LossCurveCallback.html
├── models
│   ├── performance
│   │   └── loss_curves
│   └── weights
├── notebooks
│   ├── create_sample_dataset.ipynb
│   ├── dataset_exploration.ipynb
│   └── performance_analysis.ipynb
├── pyproject.toml
├── README.md
├── reports
│   ├── compiled
│   ├── figures
│   │   ├── EDA
│   │   └── performance
│   └── main.tex
├── scripts
│   └── generate_docs.py
└── source
    ├── predict.py
    ├── train.py
    └── utils
        ├── config.py
        ├── custom_classes
        │   ├── GarbageClassifier.py
        │   ├── GarbageDataModule.py
        │   ├── __init__.py
        │   └── LossCurveCallback.py
        └── __init__.py
```
`dummy.txt` files are just placeholders so GitHub keeps the folder structure.
---
## Setup
This project uses **[uv](https://github.com/astral-sh/uv)** for dependency management.  
1. Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Sync dependencies:
```bash
uv sync
```
---
## Dataset
The notebook `notebooks/create_sample_dataset.ipynb` prepares the dataset if the `sample_dataset` folder does not exist. It:
- Downloads the [Garbage Classification Dataset](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification?resource=download).
- Creates the `sample_dataset` folder inside `data/raw`.
- Reduces dataset size for lightweight experimentation.
---
## Training
- **`source/train.py`**  
  Trains the **`GarbageClassifier`** (ResNet18 with PyTorch Lightning).
Run training with:
```bash
uv run source/train.py
```
The model checkpoint will be saved at:
```
models/weights/model_resnet18_garbage.ckpt
```
---
## Prediction
- **`source/predict.py`**  
  Loads the trained model and classifies images. Supports both single image and batch folder prediction.

### Single Image Prediction
Predict a single image:
```bash
uv run source/predict.py <path_to_image>
```
Example:
```bash
uv run source/predict.py img.jpg
```
If no path is given, the default image in `config.py` is used:
```bash
uv run source/predict.py
```

### Batch Folder Prediction
Predict all images in a folder:
```bash
uv run source/predict.py <path_to_folder>
```
Examples:
```bash
uv run source/predict.py data/test_images/
uv run source/predict.py ../new_samples/
```

The script will:
- Automatically detect all valid image files (jpg, jpeg, png, bmp, gif, tiff)
- Process them sequentially with progress indicators
- Display a summary table with all predictions

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.tif`

---
## Framework
This project is built with:
- **PyTorch Lightning** for modular training.
- **ResNet18** as the backbone model for garbage classification.
---
## Documentation
- **`docs/`**: Auto-generated HTML documentation using pdoc3.
- **`scripts/generate_docs.py`**: Script to regenerate documentation from source code docstrings.

To regenerate documentation:
```bash
uv run scripts/generate_docs.py
```
---
## Reports
- **`reports/main.tex`**: Main LaTeX report with project methodology and results.  
- **`reports/figures/`**: Stores plots and visualizations.
  - `EDA/`: Exploratory data analysis figures.
  - `performance/`: Model evaluation metrics and confusion matrices.
- **`reports/compiled/`**: Compiled PDF reports.
---
## Notebooks
- **`create_sample_dataset.ipynb`**: Scripted dataset preparation and sampling.  
- **`dataset_exploration.ipynb`**: Exploratory Data Analysis (EDA) with class distribution and sample visualization.  
- **`performance_analysis.ipynb`**: Model evaluation, confusion matrices, and error analysis.
---
## Models
- **`models/weights/`**: Stores trained model checkpoints (`.ckpt` files).  
- **`models/performance/loss_curves/`**: Training and validation loss curves with metrics in JSON format.
---
## Data Organization
- **`data/raw/`**: Original unprocessed datasets.
- **`data/interim/`**: Intermediate data transformations.
- **`data/processed/`**: Final preprocessed data ready for training.
---
## Configuration
- **`source/utils/config.py`**: Centralized configuration file containing:
  - Dataset paths and class names
  - Model hyperparameters (batch size, learning rate, epochs)
  - Training and validation split ratios
