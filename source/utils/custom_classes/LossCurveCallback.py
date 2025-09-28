#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from utils import config as cfg


class LossCurveCallback(Callback):
    def __init__(self, save_dir=cfg.LOSS_CURVES_PATH):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    # ---------- Train loss per epoch ----------
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.train_losses.append(metrics["train_loss"].item())

    # ---------- Val loss and acc per epoch ----------
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.val_losses.append(metrics["val_loss"].item())
        if "val_acc" in metrics:
            self.val_accs.append(metrics["val_acc"].item())

    def on_train_end(self, trainer, pl_module):
        # ---------- Save curves as a PNG ----------
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        if len(self.val_losses) > 0:
            plt.plot(self.val_losses, label="Val Loss")
        plt.legend()
        plt.title("Loss Curves")
        plt.xlabel("Steps / Epochs")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.save_dir, "loss_curve.png"))
        plt.close()

        if len(self.val_accs) > 0:
            plt.figure()
            plt.plot(self.val_accs, label="Val Accuracy")
            plt.legend()
            plt.title("Validation Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.savefig(os.path.join(self.save_dir, "val_acc_curve.png"))
            plt.close()

        # ---------- Save raw data ----------
        data = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
        }

        with open(os.path.join(self.save_dir, "metrics.json"), "w") as f:
            json.dump(data, f)
