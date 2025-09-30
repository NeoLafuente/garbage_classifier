#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__docformat__ = "numpy"

"""
PyTorch Lightning Callback for Loss and Accuracy Curve Visualization.

This module provides a custom callback that tracks and visualizes training
and validation metrics during model training, saving plots and raw data to
disk.
"""

import os
import json
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from utils import config as cfg


class LossCurveCallback(Callback):
    """
    PyTorch Lightning callback for tracking and plotting loss curves.

    This callback monitors training loss, validation loss, and validation
    accuracy throughout the training process. At the end of training, it
    generates and saves visualization plots and raw metric data.

    Attributes
    ----------
    save_dir : str
        Directory path where plots and metrics will be saved.
    train_losses : list of float
        Training loss values collected at the end of each training epoch.
    val_losses : list of float
        Validation loss values collected at the end of each validation epoch.
    val_accs : list of float
        Validation accuracy values collected at the end of each validation
        epoch.

    Examples
    --------
    >>> from pytorch_lightning import Trainer
    >>> callback = LossCurveCallback(save_dir="./outputs/curves")
    >>> trainer = Trainer(callbacks=[callback])
    >>> trainer.fit(model, datamodule=data_module)

    Notes
    -----
    The callback creates three output files in the save_dir:
    - loss_curve.png: Plot of training and validation losses
    - val_acc_curve.png: Plot of validation accuracy
    - metrics.json: Raw metric data in JSON format
    """

    def __init__(self, save_dir=cfg.LOSS_CURVES_PATH):
        """
        Initialize the LossCurveCallback.

        Parameters
        ----------
        save_dir : str, optional
            Directory path where plots and metrics will be saved
            (default is cfg.LOSS_CURVES_PATH).

        Notes
        -----
        The save directory is created automatically if it does not exist.
        """

        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    # ---------- Train loss per epoch ----------
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch to collect training loss.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The LightningModule being trained.

        Notes
        -----
        Extracts the 'train_loss' metric from trainer.callback_metrics and
        appends it to the train_losses list.
        """

        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.train_losses.append(metrics["train_loss"].item())

    # ---------- Val loss and acc per epoch ----------
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each validation epoch to collect validation
        metrics.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The LightningModule being validated.

        Notes
        -----
        Extracts 'val_loss' and 'val_acc' metrics from trainer.callback_metrics
        and appends them to their respective lists.
        """

        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.val_losses.append(metrics["val_loss"].item())
        if "val_acc" in metrics:
            self.val_accs.append(metrics["val_acc"].item())

    def on_train_end(self, trainer, pl_module):
        """
        Called at the end of training to generate and save plots and metrics.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The trained LightningModule.

        Notes
        -----
        This method performs three main tasks:
        1. Generates and saves a loss curve plot (loss_curve.png) showing
           training loss and validation loss over epochs.
        2. Generates and saves a validation accuracy plot (val_acc_curve.png)
           if validation accuracy was tracked.
        3. Saves all raw metric data to a JSON file (metrics.json) for later
           analysis or reproduction.

        All output files are saved to the directory specified in save_dir.
        """

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
