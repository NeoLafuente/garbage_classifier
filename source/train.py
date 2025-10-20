#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for Garbage Classification Model.

This script orchestrates the training process for a garbage classification
model using PyTorch Lightning. It can be used both as a standalone script
and as an importable module.

Usage
-----
Command line:
    $ uv run source/train.py

As a module:
    from source.train import train_model
    train_model(batch_size=32, lr=1e-3, max_epochs=10)
"""
__docformat__ = "numpy"

import pytorch_lightning as pl
from pathlib import Path
from .utils import config as cfg
from .utils.custom_classes.GarbageDataModule import GarbageDataModule
from .utils.custom_classes.GarbageClassifier import GarbageClassifier
from .utils.custom_classes.LossCurveCallback import LossCurveCallback


def train_model(
    batch_size: int = 32,
    lr: float = 1e-3,
    max_epochs: int = None,
    model_save_path: str = None,
    loss_curves_dir: str = None,
    progress_callback=None
):
    """
    Train the garbage classification model.
    
    Parameters
    ----------
    batch_size : int, default=32
        Batch size for training
    lr : float, default=1e-3
        Learning rate
    max_epochs : int, optional
        Maximum number of epochs. If None, uses cfg.MAX_EPOCHS
    model_save_path : str, optional
        Path to save the trained model. If None, uses cfg.MODEL_PATH
    loss_curves_dir : str, optional
        Directory to save loss curves. If None, uses cfg.LOSS_CURVES_PATH
    progress_callback : callable, optional
        Callback function to report progress (for UI updates)
        
    Returns
    -------
    tuple
        (trainer, model, data_module) - The trained components
    """
    # Use config defaults if not provided
    if max_epochs is None:
        max_epochs = cfg.MAX_EPOCHS
    if model_save_path is None:
        model_save_path = cfg.MODEL_PATH
    if loss_curves_dir is None:
        loss_curves_dir = cfg.LOSS_CURVES_PATH
    
    # Initialize data module
    if progress_callback:
        progress_callback("Initializing data module...")
    data_module = GarbageDataModule(batch_size=batch_size)
    data_module.setup()
    
    # Initialize model
    if progress_callback:
        progress_callback("Creating model...")
    model = GarbageClassifier(num_classes=data_module.num_classes, lr=lr)
    
    # Setup callback
    loss_curve_callback = LossCurveCallback(save_dir=loss_curves_dir)
    
    # Configure trainer
    if progress_callback:
        progress_callback(f"Starting training for {max_epochs} epochs...")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[loss_curve_callback],
        num_sanity_val_steps=0
    )
    
    # Train
    trainer.fit(model, datamodule=data_module)
    
    # Save model
    if progress_callback:
        progress_callback("Saving model...")
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(model_save_path)
    
    if progress_callback:
        progress_callback(f"✅ Training complete! Model saved at {model_save_path}")
    
    print(f"Model saved at {model_save_path}")
    
    return trainer, model, data_module


# ========================
# CLI Entry Point
# ========================
if __name__ == "__main__":
    """
    Main entry point for the training script when run from command line.
    Uses default configuration from config module.
    """
    print("Starting training with default configuration...")
    train_model()