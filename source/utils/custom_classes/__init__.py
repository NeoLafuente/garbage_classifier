"""
Custom PyTorch Lightning classes for Garbage Classification.

This module contains custom implementations of:
- GarbageClassifier: ResNet18-based classifier
- GarbageDataModule: Data loading and preprocessing
- LossCurveCallback: Training metrics visualization
"""

from .GarbageClassifier import GarbageClassifier
from .GarbageDataModule import GarbageDataModule
from .LossCurveCallback import LossCurveCallback

__all__ = [
    'GarbageClassifier',
    'GarbageDataModule',
    'LossCurveCallback',
]
