"""
Custom PyTorch Lightning classes for Garbage Classification.

This module contains custom implementations of:
- GarbageClassifier: ResNet18-based classifier
- GarbageDataModule: Data loading and preprocessing
- LossCurveCallback: Training metrics visualization
"""

from source.utils.custom_classes.GarbageClassifier import GarbageClassifier
from source.utils.custom_classes.GarbageDataModule import GarbageDataModule
from source.utils.custom_classes.LossCurveCallback import LossCurveCallback

__all__ = [
    "GarbageClassifier",
    "GarbageDataModule",
    "LossCurveCallback",
]
