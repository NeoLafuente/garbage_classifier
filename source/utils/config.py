#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module for Garbage Classification Project.

This module contains all configuration parameters and constants used
throughout the garbage classification project, including dataset paths,
model parameters, and class definitions.

Examples
--------
>>> from utils import config as cfg
>>> model = GarbageClassifier(num_classes=cfg.NUM_CLASSES)
>>> trainer = pl.Trainer(max_epochs=cfg.MAX_EPOCHS)

Notes
-----
All paths are relative to the project root directory. Ensure the directory
structure matches the configured paths before running training or inference.
"""
__docformat__ = "numpy"

DATASET_PATH = '../data/raw/sample_dataset/'
"""Path to the raw garbage dataset directory containing training images."""

LOSS_CURVES_PATH = '../models/performance/loss_curves/'
"""Directory path where training/validation loss and accuracy curves are
    saved."""

MODEL_PATH = '../models/weights/model_resnet18_garbage.ckpt'
"""Path where the trained model checkpoint is saved or loaded from."""

SAMPLE_IMG_PATH = 'sample.jpg'
"""Path to a sample image used for default predictions."""

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
"""List of garbage category names for classification.
    Categories: cardboard, glass, metal, paper, plastic, trash."""

MAX_EPOCHS = 10
"""Maximum number of training epochs."""

NUM_CLASSES = len(CLASS_NAMES)
"""Number of classification categories (derived from CLASS_NAMES length)."""
