# Project Structure and Usage Guide

## Project Organization

```
.
├── data
│   ├── interim
│   ├── processed
│   └── raw
│       ├── Garbage_Dataset_Classification
│       │   └── images
│       │       ├── cardboard
│       │       ├── glass
│       │       ├── metal
│       │       ├── paper
│       │       ├── plastic
│       │       └── trash
│       └── sample_dataset
│           ├── cardboard
│           ├── glass
│           ├── metal
│           ├── paper
│           ├── plastic
│           └── trash
├── models
│   ├── performance
│   │   └── loss_curves
│   └── weights
│       └── model_resnet18_garbage.ckpt
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
└── source
    ├── predict.py
    ├── train.py
    └── utils
        ├── config.py
        └── custom_classes
            ├── GarbageClassifier.py
            ├── GarbageDataModule.py
            └── LossCurveCallback.py
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
  Loads the trained model and classifies an image.

Usage:

```bash
uv run source/predict.py <path_to_image>
```

Examples:

```bash
uv run source/predict.py img.jpg
```

If no path is given, the default image in `config.py` is used.

---

## Framework

This project is built with:

- **PyTorch Lightning** for modular training.
- **ResNet18** as the backbone model for garbage classification.

---

## Reports

- **`reports/main.tex`**: Main LaTeX report.  
- **`reports/figures`**: Stores plots from EDA and model performance.  
- **`reports/compiled`**: Place to store compiled report PDFs.  

---

## Notebooks

- **`dataset_exploration.ipynb`**: Exploratory Data Analysis (EDA).  
- **`performance_analysis.ipynb`**: Model evaluation and error analysis.  
- **`create_sample_dataset.ipynb`**: Scripted dataset preparation.  

---

## Models

- **`models/weights`**: Stores trained checkpoints.  
- **`models/performance/loss_curves`**: Training/validation loss curves.  