# 🗑️ Garbage Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A deep learning-based waste classification system using ResNet18 and PyTorch Lightning**

[Features](#-features) • [Installation](#-setup) • [Usage](#-usage) • [Documentation](#-documentation) • [Results](#-results)

</div>

---

## 📋 Overview

This project implements an image classification system to automatically categorize waste materials into six classes: **cardboard**, **glass**, **metal**, **paper**, **plastic**, and **trash**. Built with PyTorch Lightning and ResNet18, it provides a modular, scalable solution for waste management automation.

### 🎯 Key Features

- ✨ **ResNet18 backbone** fine-tuned for garbage classification
- ⚡ **PyTorch Lightning** framework for clean, scalable training
- 📊 **Comprehensive EDA** with visualization notebooks
- 🔄 **Batch and single-image prediction** support
- 📈 **Automatic metrics tracking** with loss curves and performance reports
- 📚 **Auto-generated documentation** using pdoc3
- 🎓 **LaTeX report generation** for academic documentation

---

## 🚀 Quick Start
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Activate uv
source $HOME/.local/bin/env
```

### Interactive demo
```bash
# Create Virtual Environment
uv venv garbage-env --python 3.12

# Install PyPI package
uv pip install -U -i https://pypi.org/simple/ garbage-classifier

# Launch demo
garbage-app
```

### Full project
```bash
# Clone the repository
git clone https://github.com/NeoLafuente/garbage_classifier.git
cd garbage_classifier

# Sync dependencies
uv sync

# Train the model
uv run source/train.py

# Make a prediction
uv run source/predict.py path/to/image.jpg
```

---

## 📦 Dataset

The project uses the [Garbage Classification Dataset](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification?resource=download) from Kaggle.

### Dataset Preparation

The notebook `notebooks/create_sample_dataset.ipynb` automatically prepares the dataset:

- Downloads the Garbage Classification Dataset
- Creates the `sample_dataset` folder inside `data/raw`
- Reduces dataset size for lightweight experimentation

**Dataset Structure:**
- **6 classes**: cardboard, glass, metal, paper, plastic, trash
- **Location**: `data/raw/Garbage_Dataset_Classification/`
- **Metadata**: Class distributions and image statistics in `metadata.csv`

---

## 💻 Usage

### Training

Train the **GarbageClassifier** model (ResNet18 with PyTorch Lightning):

```bash
uv run source/train.py
```

**Output:**
- Model checkpoint: `models/weights/model_resnet18_garbage.ckpt`
- Loss curves: `models/performance/loss_curves/`
- Training logs with metrics (accuracy, precision, recall, F1-score)

**Configuration:**
Edit `source/utils/config.py` to customize:
- Batch size
- Learning rate
- Number of epochs
- Train/validation split ratio

---

### Prediction

Load the trained model and classify new images.

#### 📸 Single Image Prediction

```bash
# Predict a specific image
uv run source/predict.py img.jpg

# Use default image from config
uv run source/predict.py
```

#### 📁 Batch Folder Prediction

Process all images in a folder:

```bash
uv run source/predict.py data/test_images/
uv run source/predict.py ../new_samples/
```

**Features:**
- Auto-detects valid image files (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.tif`)
- Progress indicators for batch processing
- Summary table with all predictions

**Example Output:**
```
Predicting: cardboard_sample.jpg
Class: cardboard | Confidence: 98.5%
```

---

## 🏗️ Model Architecture

### GarbageClassifier

- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Framework**: PyTorch Lightning
- **Input**: 224x224 RGB images
- **Output**: 6-class probability distribution

### Custom Components

1. **GarbageDataModule**: PyTorch Lightning DataModule for efficient data loading
2. **LossCurveCallback**: Custom callback for tracking and saving loss curves
3. **GarbageClassifier**: Main model class with training/validation logic

---

## 📚 Documentation

### Auto-generated Documentation

HTML documentation is auto-generated from source code docstrings using pdoc3.

**View Documentation:**
- Open `docs/index.html` in your browser

**Regenerate Documentation:**
```bash
uv run scripts/generate_docs.py
```

---

## 📓 Notebooks

Interactive Jupyter notebooks for exploration and analysis:

| Notebook | Description |
|----------|-------------|
| `create_sample_dataset.ipynb` | Dataset preparation and sampling |
| `dataset_exploration.ipynb` | EDA with class distribution and visualizations |
| `performance_analysis.ipynb` | Model evaluation, confusion matrices, error analysis |

---

## 📄 Reports

### LaTeX Report

Academic-style report with methodology and results:

- **Source**: `reports/main.tex`
- **Compiled PDF**: `reports/compiled/`
- **Figures**: 
  - `reports/figures/EDA/` - Exploratory data analysis
  - `reports/figures/performance/` - Model metrics and evaluation

---

## ⚙️ Configuration

Central configuration in `source/utils/config.py`:

```python
# Dataset configuration
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
DATA_PATH = 'data/raw/Garbage_Dataset_Classification'

# Model hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

# Data split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
```

---

## 📁 Project Organization

```
.
├── data
│   ├── interim              # Intermediate data transformations
│   ├── processed            # Final preprocessed data
│   └── raw                  # Original unprocessed datasets
│       └── Garbage_Dataset_Classification
│           ├── images       # Image files by class
│           └── metadata.csv # Dataset statistics
├── docs                     # Auto-generated HTML documentation
├── models
│   ├── performance          # Loss curves and metrics
│   │   └── loss_curves
│   └── weights              # Trained model checkpoints (.ckpt)
├── notebooks                # Jupyter notebooks for EDA and analysis
│   ├── create_sample_dataset.ipynb
│   ├── dataset_exploration.ipynb
│   └── performance_analysis.ipynb
├── reports                  # LaTeX reports and figures
│   ├── compiled             # PDF reports
│   ├── figures
│   │   ├── EDA              # Exploratory analysis plots
│   │   └── performance      # Model evaluation plots
│   └── main.tex             # Main report source
├── scripts                  # Utility scripts
│   └── generate_docs.py     # Documentation generator
├── source                   # Main source code
│   ├── predict.py           # Prediction script
│   ├── train.py             # Training script
│   └── utils
│       ├── config.py        # Configuration file
│       └── custom_classes   # Model implementations
│           ├── GarbageClassifier.py
│           ├── GarbageDataModule.py
│           └── LossCurveCallback.py
├── pyproject.toml           # Project dependencies (uv)
└── README.md
```

*Note: `dummy.txt` files are placeholders to preserve empty folder structure in Git.*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Garbage Classification Dataset](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification) by zlatan599 on Kaggle
- **Framework**: [PyTorch Lightning](https://lightning.ai/) for the excellent deep learning framework
- **Model**: ResNet18 architecture from [torchvision](https://pytorch.org/vision/stable/index.html)

---

## 📧 Contact

**Neo Lafuente** - [@NeoLafuente](https://github.com/NeoLafuente)

Project Link: [https://github.com/NeoLafuente/garbage_classifier](https://github.com/NeoLafuente/garbage_classifier)

---

<div align="center">
Made with ❤️ and ♻️ for a cleaner planet
</div>