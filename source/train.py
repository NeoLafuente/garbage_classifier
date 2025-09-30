#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__docformat__ = "numpy"

"""
Training Script for Garbage Classification Model.

This script orchestrates the training process for a garbage classification
model using PyTorch Lightning. It initializes the data module, model,
callbacks, and trainer, then executes the training loop and saves the
trained model checkpoint.

The script performs the following steps:
1. Initializes the GarbageDataModule with stratified train/test split
2. Creates a GarbageClassifier model (ResNet18-based)
3. Sets up loss curve visualization callback
4. Configures PyTorch Lightning Trainer with specified hyperparameters
5. Trains the model on the garbage dataset
6. Saves the trained model checkpoint

Usage
-----
Command line::

    uv run train.py

Notes
-----
Configuration parameters are loaded from `utils.config` module:
- `MAX_EPOCHS`: Maximum number of training epochs
- `LOSS_CURVES_PATH`: Directory for saving loss and accuracy plots
- `MODEL_PATH`: Path where the trained model checkpoint will be saved

The training uses automatic device selection (GPU if available, otherwise CPU)
and disables sanity validation steps for faster startup.
"""


import pytorch_lightning as pl
from utils import config as cfg
from utils.custom_classes.GarbageDataModule import GarbageDataModule
from utils.custom_classes.GarbageClassifier import GarbageClassifier
from utils.custom_classes.LossCurveCallback import LossCurveCallback


# ========================
# Training
# ========================
if __name__ == "__main__":
    data_module = GarbageDataModule(batch_size=32)
    data_module.setup()

    model = GarbageClassifier(num_classes=data_module.num_classes, lr=1e-3)

    loss_curve_callback = LossCurveCallback(save_dir=cfg.LOSS_CURVES_PATH)

    trainer = pl.Trainer(
        max_epochs=cfg.MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[loss_curve_callback],
        num_sanity_val_steps=0
    )
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(cfg.MODEL_PATH)
    print(f"Model saved at {cfg.MODEL_PATH}")
