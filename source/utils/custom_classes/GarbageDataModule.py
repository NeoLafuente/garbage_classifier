#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from sklearn.model_selection import train_test_split
import numpy as np
from utils import config as cfg

class GarbageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    
    def setup(self, stage=None):
        # Load full dataset
        full_dataset = datasets.ImageFolder(cfg.DATASET_PATH, transform=self.transform)
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

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self): # TODO: change to val
        return DataLoader(self.test_dataset, batch_size=1000, shuffle=False, num_workers=self.num_workers)