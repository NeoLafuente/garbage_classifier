# Project Structure and Usage Guide

## Project Organization

```
.
├── data
│   ├── interim
│   │   └── dummy.txt
│   ├── processed
│   │   └── dummy.txt
│   └── raw
│       ├── dummy.txt
│       └── sample_dataset
│           ├── cardboard
│           ├── glass
│           ├── metal
│           ├── paper
│           ├── plastic
│           └── trash
├── models
│   ├── dummy.txt
│   └── model_resnet18_garbage.ckpt
├── notebooks
│   └── create_sample_dataset.ipynb
├── pyproject.toml
├── README.md
├── reports
│   └── figures
│       └── dummy.txt
├── source
│   ├── predict.py
│   ├── train.py
│   └── utils
│       ├── config.py
│       └── custom_classes
│           ├── GarbageClassifier.py
│           └── GarbageDataModule.py
├── tree.txt
└── uv.lock
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

The notebooks/create_sample_dataset.ipynb notebook is used to set up the dataset if the sample_dataset folder 
does not exist. So, This notebook:
- Downloads the [Garbage Classification Dataset](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification?resource=download) from Kaggle.
- Creates the `sample_dataset` folder inside `data/raw`.
- Reduces the dataset size (fewer images) for easier experimentation.
 
---

## Training

- **`source/train.py`**  
  Trains the **`GarbageClassifier`**, which is a **ResNet18** model implemented in PyTorch Lightning.  

Run training with:

```bash
uv run source/train.py
```

After training, the model checkpoint will be saved in:

```
models/model_resnet18_garbage.ckpt
```

---

## Prediction

- **`source/predict.py`**  
  Loads the trained model (`model_resnet18_garbage.ckpt`) and classifies an image.

Usage:

```bash
uv run source/predict.py <path_to_image>
```

Examples:

```bash
uv run source/predict.py img.jpg
```

If no argument is provided, it will use the sample image path defined in `config.py`.

---

## Framework

This project is built with:

- **[PyTorch Lightning](https://www.pytorchlightning.ai/)** for training and organizing the deep learning pipeline.
- **ResNet18** as the backbone model for garbage classification.

---