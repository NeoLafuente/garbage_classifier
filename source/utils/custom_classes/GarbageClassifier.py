#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__docformat__ = "numpy"

"""
Garbage Classification Model Module.

This module implements a PyTorch Lightning module for garbage classification
using a pretrained ResNet18 model. The classifier is fine-tuned for a 6-class
garbage classification problem (cardboard, glass, metal, paper, plastic,
trash).

The model uses transfer learning by freezing the pretrained ResNet18 feature
extraction layers and training only the final classification layer.
"""

import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models


# ========================
# LightningModule
# ========================
class GarbageClassifier(pl.LightningModule):
    """
    Pretrained (ImageNet) ResNet18 adapted to Garbage Dataset Classification
    problem.
    It considers 6 classes: cardboard, glass, metal, paper, plastic and trash.

    Atributes
    ---------
    model : torchvision.models.resnet18
        Pretrained ResNet18 model.
    loss_fn : torch.nn.CrossEntropyLoss
        Cross entropy loss function.

    Examples
    --------
    >>> model = GarbageClassifier(num_classes=6, lr=1e-3)
    >>> trainer = pl.Trainer(max_epochs=10, accelerator="auto")
    >>> trainer.fit(model, datamodule=data_module)
    """

    def __init__(self, num_classes, lr=1e-3):
        """
        Initialize the GarbageClassifier model.

        Parameters
        ----------
        num_classes : int
            Number of output classes for classification.
        lr : float, optional
            Learning rate for the optimizer (default is 1e-3).
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Freeze feature layers
        for param in self.model.parameters():
            param.requires_grad = False

        # New classifier layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of images with shape (batch_size, channels, height,
            width).

        Returns
        -------
        torch.Tensor
            Output logits with shape (batch_size, num_classes).
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Model parameters are updated according to the classification error
        of a subset of train images.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Subset (batch) of images coming from the train dataloader.
            Contains input images and corresponding labels.
        batch_idx : int
            Identifier of the batch within the current epoch.

        Returns
        -------
        torch.Tensor
            Classification error (cross entropy loss) of trained image batch.
        """
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        self.log('train_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Compute validation loss and accuracy for a batch of validation images.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            Subset (batch) of images coming from the validation dataloader.
            Contains input images and corresponding labels.
        batch_idx : int
            Identifier of the batch within the current validation epoch.

        Returns
        -------
        torch.Tensor
            Validation accuracy for the current batch.
        """
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        preds = out.argmax(dim=1)
        acc = (preds == yb).float().mean()
        self.log('val_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False)
        self.log('val_acc',
                 acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        return acc

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns
        -------
        torch.optim.Adam
            Adam optimizer configured with model parameters and learning rate
            from hyperparameters.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
