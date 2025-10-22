#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

from source.utils.custom_classes.GarbageClassifier import GarbageClassifier
from source.utils.custom_classes.GarbageDataModule import GarbageDataModule
from source.utils import config as cfg


class GarbageModelAnalyzer:
    def __init__(self, dataset_path=None, performance_path=None):
        self.dataset_path = dataset_path or os.path.join("..", "data", "raw", "sample_dataset")
        self.performance_path = performance_path or '../reports/figures/performance/'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metadata_path = Path(cfg.DATASET_PATH).parent / "metadata.csv"
        
        if metadata_path.exists():
            self.df = pd.read_csv(metadata_path)
        else:
            print(f"Warning: metadata.csv not found at {metadata_path}")
            self.df = None
        
        self.model = None
        self.data_module = None

    def load_model(self, checkpoint_path=None, num_classes=None):
        checkpoint_path = checkpoint_path or cfg.MODEL_PATH
        num_classes = num_classes or cfg.NUM_CLASSES
        print("Loading model...")
        self.model = GarbageClassifier.load_from_checkpoint(checkpoint_path, num_classes=num_classes)
        self.model.to(self.device).eval()
        print("Model loaded.")

    def setup_data(self, batch_size=32):
        self.data_module = GarbageDataModule(batch_size=batch_size)
        self.data_module.setup()
        file_names = []
        for root, dirs, files in os.walk(cfg.DATASET_PATH):
            for file in files:
                file_names.append(file)
        self.df_subset = self.df[self.df['filename'].isin(file_names)].reset_index(drop=True).copy()

    def evaluate_loader(self, loader):
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
        num_classes = self.data_module.num_classes
        cm = confusion_matrix(labels, preds, labels=range(num_classes))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cfg.CLASS_NAMES)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {set_name} set")
        plt.savefig(os.path.join(self.performance_path, f"confusion_mat_{set_name.lower()}.pdf"), dpi=80)
        plt.show()

        # Normalized
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=cfg.CLASS_NAMES)
        disp_norm.plot(cmap=plt.cm.Blues)
        plt.title(f"Normalized Confusion Matrix - {set_name} set")
        plt.savefig(os.path.join(self.performance_path, f"confusion_mat_{set_name.lower()}_norm.pdf"), dpi=80)
        plt.show()

        # TP, FP, FN, TN
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)
        for i in range(num_classes):
            print(f"Clase {i}: TP={TP[i]}, FP={FP[i]}, FN={FN[i]}, TN={TN[i]}")

    def plot_top_misclassified(self, df_set, y_true, y_pred, y_proba, N=10, filename=None):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Handle case: y_true are integers vs strings
        if np.issubdtype(y_true.dtype, np.integer):
            true_indices = y_true
            classes = sorted(df_set['label'].unique())
        else:
            classes = sorted(np.unique(y_true))
            class_to_idx = {cls: i for i, cls in enumerate(classes)}
            true_indices = np.array([class_to_idx[label] for label in y_true])

        true_confidences = y_proba[np.arange(len(y_true)), true_indices]
        misclassified_idx = np.where(y_true != y_pred)[0]

        if len(misclassified_idx) == 0:
            print("No misclassified samples found!")
            return

        sorted_idx = misclassified_idx[np.argsort(true_confidences[misclassified_idx])]
        selected_idx = sorted_idx[:N]

        plt.figure(figsize=(15, 3 * (N // 5 + 1)))
        for i, idx in enumerate(selected_idx, 1):
            row = df_set.iloc[idx]
            img_path = os.path.join(self.dataset_path, row['label'], row['filename'])
            if not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except:
                continue
            plt.subplot(int(np.ceil(N/5)), 5, i)
            plt.imshow(img)
            plt.axis("off")
            plt.title(
                f"True: {row['label']}\nPred: {classes[y_pred[idx]]}\nConf True Class: {true_confidences[idx]:.2f}",
                fontsize=9,
                color="red"
            )
        plt.tight_layout()
        if filename:
            plt.savefig(os.path.join(self.performance_path, f"{filename}.pdf"), dpi=80)
        plt.show()

    def plot_calibration_curves(self, y_true, y_probs):
        num_classes = self.data_module.num_classes
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.cpu().numpy()
        else:
            y_true_np = y_true
        
        # y_probs ya debería estar en numpy (desde evaluate_loader)
        # pero por si acaso:
        if isinstance(y_probs, torch.Tensor):
            y_probs = y_probs.cpu().numpy()
        
        for c in range(num_classes):
            ax = axes[c]
            y_true_c = (y_true_np == c).astype(int)
            y_prob_c = y_probs[:, c]
            
            frac_pos, mean_pred = calibration_curve(y_true_c, y_prob_c, n_bins=10)
            ax.plot(mean_pred, frac_pos, marker='o', label=f'Class {c}')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Reference')
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
        # plt.savefig(os.path.join(self.performance_path, "calibration_curves.pdf"), dpi=80)
        plt.show()