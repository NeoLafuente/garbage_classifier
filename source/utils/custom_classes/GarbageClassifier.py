#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models

# ========================
# LightningModule
# ========================
class GarbageClassifier(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze feature layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # New classifier layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        out = self(xb)
        preds = out.argmax(dim=1)
        acc = (preds == yb).float().mean()
        self.log('val_acc', acc, prog_bar=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)