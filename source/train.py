#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
from utils import config as cfg
from utils.custom_classes.GarbageDataModule import GarbageDataModule
from utils.custom_classes.GarbageClassifier import GarbageClassifier


# ========================
# Training
# ========================
if __name__ == "__main__":
    data_module = GarbageDataModule(batch_size=32)
    data_module.setup()
    
    model = GarbageClassifier(num_classes=data_module.num_classes, lr=1e-3)

    trainer = pl.Trainer(max_epochs=3, accelerator="auto", devices=1)
    trainer.fit(model, datamodule=data_module)
    
    # Guardar modelo
    # torch.save(model.model.state_dict(), cfg.MODEL_PATH)
    
    trainer.save_checkpoint(cfg.MODEL_PATH)
    print(f"Model saved at {cfg.MODEL_PATH}")
