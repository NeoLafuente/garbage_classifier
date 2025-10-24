#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

from source.utils.custom_classes.GarbageClassifier import GarbageClassifier
from source.utils.custom_classes.GarbageDataModule import GarbageDataModule
from source.utils import config as cfg
from source.utils.config import get_valid_dir as gvd


class GarbageModelAnalyzer:
    """
    Analyzer class for evaluating and visualizing garbage
    classification model performance.

    This class provides comprehensive tools for model evaluation
    including confusion matrices, calibration curves, and misclassified
    sample visualization. It handles both model loading and data
    module setup, with support for GPU acceleration.

    Attributes
    ----------
    dataset_path : str
        Path to the dataset folder containing images organized by class.
    performance_path : str
        Path to the directory where performance figures are saved.
    device : torch.device
        Computational device (CUDA if available, otherwise CPU).
    df : pd.DataFrame or None
        Metadata DataFrame containing dataset information.
    model : GarbageClassifier or None
        Loaded model instance for inference.
    data_module : GarbageDataModule or None
        Data module for dataset loading and preprocessing.
    """

    def __init__(self, dataset_path=None, performance_path=None):
        """
        Initialize the GarbageModelAnalyzer instance.

        Loads metadata from the dataset path and sets up paths
        for storing performance figures. Detects CUDA availability
        for GPU acceleration.

        Parameters
        ----------
        dataset_path : str, optional
            Path to the dataset folder. Default is
            "../data/raw/sample_dataset".
        performance_path : str, optional
            Path to the directory for saving performance figures. Default is
            "../reports/figures/performance/".

        Returns
        -------
        None
        """
        if dataset_path is None:
            dataset_path = cfg.SAMPLE_DATASET_PATH
        self.dataset_path = gvd(dataset_path)

        # os.path.join(
        #     "..", "data", "raw", "sample_dataset"
        # )

        if performance_path is None:
            performance_path = "../reports/figures/performance/"
        self.performance_path = gvd(performance_path)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        metadata_path = Path(f"{gvd(Path(cfg.DATASET_PATH).parent)}/metadata.csv")

        if metadata_path.exists():
            self.df = pd.read_csv(metadata_path)
        else:
            print(f"Warning: metadata.csv not found at {metadata_path}")
            self.df = None

        self.model = None
        self.data_module = None

    def load_model(self, checkpoint_path=None, num_classes=None):
        """
        Load a trained GarbageClassifier model from a checkpoint.

        Loads a model from a PyTorch Lightning checkpoint and moves it to the
        specified device in evaluation mode.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to the model checkpoint file. If None, uses cfg.MODEL_PATH.
            Default is None.
        num_classes : int, optional
            Number of output classes. If None, uses cfg.NUM_CLASSES.
            Default is None.

        Returns
        -------
        None

        Notes
        -----
        Model is automatically set to evaluation mode and moved
        to the configured device.
        """
        if checkpoint_path is None:
            checkpoint_path = cfg.MODEL_PATH
        checkpoint_path = (
            f"{gvd(str(Path(checkpoint_path).parent))}"
            f"/"
            f"{cfg.MODEL_PATH.split('/')[-1]}"
        )
        num_classes = num_classes or cfg.NUM_CLASSES
        print("Loading model...")
        self.model = GarbageClassifier.load_from_checkpoint(
            checkpoint_path, num_classes=num_classes
        )
        self.model.to(self.device).eval()
        print("Model loaded.")

    def setup_data(self, batch_size=32):
        """
        Set up the data module and filter metadata for available samples.

        Initializes the GarbageDataModule and creates a filtered subset
        of metadata containing only files present in the dataset directory.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for data loading. Default is 32.

        Returns
        -------
        None

        Notes
        -----
        Creates self.df_subset containing only samples actually
        present in the dataset.
        """

        self.data_module = GarbageDataModule(batch_size=batch_size)
        self.data_module.setup()
        file_names = []
        for root, dirs, files in os.walk(gvd(cfg.DATASET_PATH)):
            for file in files:
                file_names.append(file)
        self.df_subset = (
            self.df[
                self.df["filename"].isin(file_names)
            ].reset_index(drop=True).copy()
        )

    def evaluate_loader(self, loader):
        """
        Evaluate model on a data loader and collect predictions
        and probabilities.

        Iterates through a PyTorch DataLoader, performs inference,
        and collects predictions, true labels, and confidence scores.

        Parameters
        ----------
        loader : torch.utils.data.DataLoader
            DataLoader containing batches of (images, labels).

        Returns
        -------
        tuple
            A tuple containing:
            - all_preds : torch.Tensor
                Predicted class indices.
            - all_labels : torch.Tensor
                True class labels.
            - all_probs : np.ndarray
                Confidence scores for each class (shape: [N, num_classes]).

        Notes
        -----
        Model inference is performed without gradient computation
        for efficiency.
        Probabilities are computed using softmax activation.
        """
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                out = self.model(xb)
                preds = out.argmax(dim=1)
                probs = torch.softmax(out, dim=1)
                all_preds.append(preds)
                all_probs.append(probs.cpu())
                all_labels.append(yb)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs).numpy()
        return all_preds, all_labels, all_probs

    def plot_confusion_matrix(self, labels, preds, set_name="Train"):
        """
        Plot and save confusion matrices (raw and normalized)
        with class metrics.

        Generates both raw and normalized confusion matrices,
        computes TP, FP, FN, TN per class, and saves visualizations
        as PDF files.

        Parameters
        ----------
        labels : array-like
            True class labels.
        preds : array-like
            Predicted class labels.
        set_name : str, optional
            Name of the dataset split (e.g., "Train", "Val", "Test")
            for plot titles and filenames. Default is "Train".

        Returns
        -------
        None

        Side Effects
        -----------
        - Saves confusion_mat_{set_name.lower()}.pdf to performance_path.
        - Saves confusion_mat_{set_name.lower()}_norm.pdf to performance_path.
        - Prints TP, FP, FN, TN statistics for each class.
        - Displays matplotlib figures.

        Notes
        -----
        Normalized confusion matrix divides by row sums to show percentages
        per class.
        """
        num_classes = self.data_module.num_classes
        cm = confusion_matrix(labels, preds, labels=range(num_classes))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=cfg.CLASS_NAMES
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {set_name} set")
        plt.savefig(
            os.path.join(
                self.performance_path, f"confusion_mat_{set_name.lower()}.pdf"
            ),
            dpi=80,
        )
        plt.show()

        # Normalized
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        disp_norm = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm, display_labels=cfg.CLASS_NAMES
        )
        disp_norm.plot(cmap=plt.cm.Blues)
        plt.title(f"Normalized Confusion Matrix - {set_name} set")
        plt.savefig(
            os.path.join(
                self.performance_path,
                f"confusion_mat_{set_name.lower()}_norm.pdf"
            ),
            dpi=80,
        )
        plt.show()

        # TP, FP, FN, TN
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)
        for i in range(num_classes):
            print(f"Clase {i}: TP={TP[i]}, FP={FP[i]}, FN={FN[i]}, TN={TN[i]}")

    def plot_top_misclassified(
        self, df_set, y_true, y_pred, y_proba, N=10, filename=None
    ):
        """
        Visualize the top N misclassified samples with lowest confidence.

        Identifies misclassified samples, sorts them by confidence
        on the true class, and displays the least confident (worst)
        predictions with their images.

        Parameters
        ----------
        df_set : pd.DataFrame
            DataFrame containing sample metadata with 'label' and 'filename'
            columns.
        y_true : array-like
            True class labels (can be integers or strings).
        y_pred : array-like
            Predicted class labels (integers).
        y_proba : np.ndarray
            Confidence scores matrix (shape: [N, num_classes]).
        N : int, optional
            Number of top misclassified samples to display. Default is 10.
        filename : str, optional
            Filename (without extension) to save the figure. If provided,
            saves as PDF to performance_path. Default is None (no save).

        Returns
        -------
        None

        Side Effects
        -----------
        - Displays matplotlib figure with misclassified samples.
        - Saves figure to performance_path/{filename}.pdf
        if filename is provided.

        Notes
        -----
        Samples are sorted by confidence on the true class in ascending order,
        showing the most uncertain misclassifications first.
        Images are loaded from dataset_path/{label}/{filename} structure.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Handle case: y_true are integers vs strings
        if np.issubdtype(y_true.dtype, np.integer):
            true_indices = y_true
            classes = sorted(df_set["label"].unique())
        else:
            classes = sorted(np.unique(y_true))
            class_to_idx = {cls: i for i, cls in enumerate(classes)}
            true_indices = np.array([class_to_idx[label] for label in y_true])

        true_confidences = y_proba[np.arange(len(y_true)), true_indices]
        misclassified_idx = np.where(y_true != y_pred)[0]

        if len(misclassified_idx) == 0:
            print("No misclassified samples found!")
            return

        sorted_idx = misclassified_idx[
            np.argsort(true_confidences[misclassified_idx])
        ]
        selected_idx = sorted_idx[:N]

        plt.figure(figsize=(15, 3 * (N // 5 + 1)))
        for i, idx in enumerate(selected_idx, 1):
            row = df_set.iloc[idx]
            img_path = os.path.join(
                self.dataset_path,
                row["label"],
                row["filename"]
            )
            if not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            plt.subplot(int(np.ceil(N / 5)), 5, i)
            plt.imshow(img)
            plt.axis("off")
            plt.title(
                f"True: {row['label']}\nPred: \
                    {classes[y_pred[idx]]}\nConf True Class: \
                        {true_confidences[idx]:.2f}",
                fontsize=9,
                color="red",
            )
        plt.tight_layout()
        if filename:
            plt.savefig(
                os.path.join(
                    self.performance_path,
                    f"{filename}.pdf"
                ),
                dpi=80
            )
        plt.show()

    def plot_calibration_curves(self, y_true, y_probs):
        """
        Plot calibration curves for all classes using one-vs-rest approach.

        Generates calibration curves showing the relationship between predicted
        probabilities and actual positive fractions for each class.

        Parameters
        ----------
        y_true : array-like or torch.Tensor
            True class labels (integers). Shape: [N,].
        y_probs : np.ndarray or torch.Tensor
            Predicted probability matrix. Shape: [N, num_classes].

        Returns
        -------
        None

        Side Effects
        -----------
        - Displays matplotlib figure with 2x3 grid of calibration curves.
        - One subplot per class showing calibration curve and reference
        diagonal.

        Notes
        -----
        Uses sklearn's calibration_curve function with 10 bins for each class.
        Each class is converted to a binary classification problem (OVR).
        Calibration curves below the diagonal indicate:
        -> overconfident predictions.
        Calibration curves above the diagonal indicate:
        -> underconfident predictions.
        """
        num_classes = self.data_module.num_classes
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.cpu().numpy()
        else:
            y_true_np = y_true

        if isinstance(y_probs, torch.Tensor):
            y_probs = y_probs.cpu().numpy()

        for c in range(num_classes):
            ax = axes[c]
            y_true_c = (y_true_np == c).astype(int)
            y_prob_c = y_probs[:, c]

            frac_pos, mean_pred = calibration_curve(
                y_true_c, y_prob_c, n_bins=10
            )
            ax.plot(mean_pred, frac_pos, marker="o", label=f"Class {c}")
            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color="gray",
                label="Reference"
            )
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title(f"Calibration Curve: {cfg.CLASS_NAMES[c]}")
            ax.set_xticks(np.arange(0, 1.1, 0.1))
            ax.set_yticks(np.arange(0, 1.1, 0.1))
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True)
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()
