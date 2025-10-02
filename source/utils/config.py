#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Module for Garbage Classification Project.

This module contains all configuration parameters and constants used
throughout the garbage classification project, including dataset paths,
model parameters, and class definitions.

Attributes
----------
DATASET_PATH : str
    Path to the raw garbage dataset directory containing training images.
LOSS_CURVES_PATH : str
    Directory path where training/validation loss and accuracy curves are
    saved.
MODEL_PATH : str
    Path where the trained model checkpoint is saved or loaded from.
SAMPLE_IMG_PATH : str
    Path to a sample image used for default predictions.
CLASS_NAMES : list of str
    List of garbage category names for classification.
    Categories: cardboard, glass, metal, paper, plastic, trash.
MAX_EPOCHS : int
    Maximum number of training epochs.
NUM_CLASSES : int
    Number of classification categories (derived from CLASS_NAMES length).

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
LOSS_CURVES_PATH = '../models/performance/loss_curves/'
MODEL_PATH = '../models/weights/model_resnet18_garbage.ckpt'
SAMPLE_IMG_PATH = '../data/raw/sample.jpg'
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MAX_EPOCHS = 10
NUM_CLASSES = len(CLASS_NAMES)
