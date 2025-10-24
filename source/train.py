#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for Garbage Classification Model.

This script orchestrates the training process for a garbage classification
model using PyTorch Lightning. It can be used both as a standalone script
and as an importable module. Includes carbon emissions tracking.

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
from codecarbon import EmissionsTracker
from source.utils import config as cfg
from source.utils.config import get_valid_dir as gvd
from source.utils.carbon_utils import (
    kg_co2_to_car_distance,
    format_car_distance
)
from source.utils.custom_classes.GarbageDataModule import GarbageDataModule
from source.utils.custom_classes.GarbageClassifier import GarbageClassifier
from source.utils.custom_classes.LossCurveCallback import LossCurveCallback


def train_model(
    batch_size: int = 32,
    lr: float = 1e-3,
    max_epochs: int = None,
    model_save_path: str = None,
    loss_curves_dir: str = None,
    track_carbon: bool = True,
    progress_callback=None,
):
    """
    Train the garbage classification model using PyTorch Lightning.

    Orchestrates the complete training pipeline including data loading, model
    initialization, training, and evaluation. Optionally tracks
    carbon emissions during training and provides progress callbacks
    for UI integration.

    Parameters
    ----------
    batch_size : int, default=32
        Batch size for training and validation data loaders.
    lr : float, default=1e-3
        Learning rate for the optimizer.
    max_epochs : int, optional
        Maximum number of training epochs. If None, uses cfg.MAX_EPOCHS.
        Default is None.
    model_save_path : str, optional
        Path to save the trained model checkpoint.
        If None, uses cfg.MODEL_PATH.
        Default is None.
    loss_curves_dir : str, optional
        Directory to save loss curve visualizations.
        If None, uses cfg.LOSS_CURVES_PATH.
        Default is None.
    track_carbon : bool, default=True
        Whether to track carbon emissions during training using CodeCarbon.
    progress_callback : callable, optional
        Callback function to report training progress.
        Called with a message string for UI updates.
        Default is None (no progress reporting).

    Returns
    -------
    dict
        Dictionary containing:
        - 'trainer': pl.Trainer
            PyTorch Lightning trainer instance.
        - 'model': GarbageClassifier
            Trained model instance.
        - 'data_module': GarbageDataModule
            Data module used for training.
        - 'emissions': dict or None
            Carbon emissions data with keys: 'emissions_kg', 'emissions_g',
            'car_distance_km', 'car_distance_m', 'car_distance_formatted',
            'duration_seconds'. None if track_carbon=False.
        - 'metrics': dict
            Training and validation metrics with keys: 'train_acc', 'val_acc',
            'train_loss', 'val_loss'.

    Raises
    ------
    Exception
        Any exception during training is re-raised after stopping
        emissions tracker if applicable.

    Notes
    -----
    Carbon emissions are converted to equivalent car driving distance for
    intuitive understanding. Model checkpoint is automatically saved to disk.
    If progress_callback is provided, it receives status updates at key points
    during training initialization and completion.
    """

    # create_directory_structure()

    # Use config defaults if not provided
    if max_epochs is None:
        max_epochs = cfg.MAX_EPOCHS
    if model_save_path is None:
        model_save_path = f"{gvd(cfg.WEIGHTS_DIR)}/model_resnet18_garbage.ckpt"
    if loss_curves_dir is None:
        loss_curves_dir = gvd(cfg.LOSS_CURVES_PATH)

    # Initialize emissions tracker
    emissions_data = None
    if track_carbon:
        tracker = EmissionsTracker(
            project_name="garbage_classifier_training",
            output_dir=gvd(cfg.BEST_MODEL_DIR),
            log_level="warning",  # Reduce console output
        )
        tracker.start()

    try:
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
        loss_curve_callback = LossCurveCallback(save_dir=gvd(loss_curves_dir))

        # Configure trainer
        if progress_callback:
            progress_callback(f"Starting training for {max_epochs} epochs...")
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[loss_curve_callback],
            num_sanity_val_steps=0,
        )

        # Train
        trainer.fit(model, datamodule=data_module)

        # Extract final metrics
        metrics = {}
        if trainer.callback_metrics:
            metrics["train_acc"] = trainer.callback_metrics.get(
                "train_acc",
                None
            )
            metrics["val_acc"] = trainer.callback_metrics.get(
                "val_acc",
                None
            )
            metrics["train_loss"] = trainer.callback_metrics.get(
                "train_loss",
                None
            )
            metrics["val_loss"] = trainer.callback_metrics.get(
                "val_loss",
                None
            )

            # Convert tensors to float if needed
            for key in metrics:
                if metrics[key] is not None:
                    if hasattr(metrics[key], "item"):
                        metrics[key] = metrics[key].item()

        # Save model
        if progress_callback:
            progress_callback("Saving model...")
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(model_save_path)

        # Stop emissions tracking
        if track_carbon:
            emissions_kg = tracker.stop()
            car_distances = kg_co2_to_car_distance(emissions_kg)
            emissions_data = {
                "emissions_kg": emissions_kg,
                "emissions_g": emissions_kg * 1000,
                "car_distance_km": car_distances["distance_km"],
                "car_distance_m": car_distances["distance_m"],
                "car_distance_formatted": format_car_distance(emissions_kg),
                "duration_seconds": (
                    tracker._total_duration.total_seconds()
                    if hasattr(tracker, "_total_duration")
                    else None
                ),
            }

        if progress_callback:
            msg = f"✅ Training complete! Model saved at {model_save_path}"
            if emissions_data:
                msg += (
                    f"\n🌍 Carbon footprint:  \
                        {emissions_data['emissions_g']:.2f}g \
                            CO₂eq"
                )
                msg += f"\n🚗 Equivalent to driving: \
                    {emissions_data['car_distance_formatted']}"
            progress_callback(msg)

        print(f"Model saved at {model_save_path}")
        if emissions_data:
            print(
                f"Carbon emissions: {emissions_data['emissions_kg']:.6f}kg \
                    CO₂eq ({emissions_data['emissions_g']:.2f}g)"
            )
            print(f"Equivalent to driving: \
                {emissions_data['car_distance_formatted']}")

        return {
            "trainer": trainer,
            "model": model,
            "data_module": data_module,
            "emissions": emissions_data,
            "metrics": metrics,
        }

    except Exception as e:
        if track_carbon:
            tracker.stop()
        raise e


# ========================
# CLI Entry Point
# ========================
def main():
    """
    Main entry point for the training script when run from command line.

    Executes the complete training pipeline using default configuration from
    the config module. Displays final metrics and carbon emissions statistics.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    print("Starting training with default configuration...")
    result = train_model()

    if result["emissions"]:
        print(
            f"\n🌍 Total carbon footprint: "
            f"{result['emissions']['emissions_g']:.2f}g CO₂eq"
        )
        print(
            f"🚗 Equivalent to driving: "
            f"{result['emissions']['car_distance_formatted']}"
        )

    if result["metrics"]:
        print("\n📊 Final Metrics:")
        if result["metrics"].get("train_acc") is not None:
            print(f"  Train Accuracy: {result['metrics']['train_acc']:.4f}")
        if result["metrics"].get("val_acc") is not None:
            print(f"  Validation Accuracy: {result['metrics']['val_acc']:.4f}")


if __name__ == "__main__":
    main()
