#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__docformat__ = "numpy"

"""
Garbage Dataset DataModule for PyTorch Lightning.

This module provides a LightningDataModule implementation for loading and
preparing the garbage classification dataset with stratified train/test splits.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from sklearn.model_selection import train_test_split
import numpy as np
from utils import config as cfg


class GarbageDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Garbage Classification Dataset.

    This DataModule handles loading, splitting, and creating dataloaders for
    the garbage classification dataset. It performs a stratified 90/10
    train/test split and applies ResNet18 ImageNet preprocessing transforms.

    Attributes
    ----------
    batch_size : int
        Number of samples per batch for training.
    num_workers : int
        Number of subprocesses to use for data loading.
    transform : torchvision.transforms.Compose
        Image preprocessing transforms from ResNet18 ImageNet weights.
    train_dataset : torch.utils.data.Subset
        Training dataset subset.
    test_dataset : torch.utils.data.Subset
        Test/validation dataset subset.
    train_idx : numpy.ndarray
        Indices of samples in the training set.
    test_idx : numpy.ndarray
        Indices of samples in the test set.
    num_classes : int
        Number of classes in the dataset.

    Examples
    --------
    >>> data_module = GarbageDataModule(batch_size=32, num_workers=4)
    >>> data_module.setup()
    >>> train_loader = data_module.train_dataloader()
    >>> val_loader = data_module.val_dataloader()
    """

    def __init__(self, batch_size=32, num_workers=4):
        """
        Initialize the GarbageDataModule.

        Parameters
        ----------
        batch_size : int, optional
            Number of samples per batch for training (default is 32).
        num_workers : int, optional
            Number of subprocesses to use for data loading (default is 4).
        """

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()

    def setup(self, stage=None):
        """
        Prepare the dataset by loading and splitting into train and test sets.

        Loads the full dataset from the configured path, performs a stratified
        90/10 train/test split to ensure balanced class distribution, and
        creates dataset subsets for training and validation.

        Parameters
        ----------
        stage : str, optional
            Current stage ('fit', 'validate', 'test', or 'predict').
            Not used in this implementation (default is None).

        Notes
        -----
        The dataset is split using a stratified approach with random_state=42
        for reproducibility. The split ratio is 90% training and 10% testing.
        """

        # Load full dataset
        full_dataset = datasets.ImageFolder(
            cfg.DATASET_PATH,
            transform=self.transform
        )
        targets = [label for _, label in full_dataset]
        self.num_classes = cfg.NUM_CLASSES

        # Stratified split 90/10
        train_idx, test_idx = train_test_split(
            np.arange(len(targets)),
            test_size=0.1,
            stratify=targets,
            random_state=42
        )

        self.train_dataset = Subset(full_dataset, train_idx)
        self.test_dataset = Subset(full_dataset, test_idx)
        self.train_idx = train_idx
        self.test_idx = test_idx

    def train_dataloader(self):
        """
        Create and return the training dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader for the training dataset with shuffling enabled.

        Notes
        -----
        The dataloader uses the configured batch_size and num_workers,
        and shuffles the data at each epoch.
        """

        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Create and return the validation dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            DataLoader for the validation/test dataset without shuffling.

        Notes
        -----
        The dataloader uses a fixed batch_size of 1000 for faster validation,
        with num_workers from configuration. Shuffling is disabled to ensure
        consistent validation metrics.
        """

        return DataLoader(self.test_dataset,
                          batch_size=1000,
                          shuffle=False,
                          num_workers=self.num_workers)
