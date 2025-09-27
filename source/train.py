#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pytorch_lightning as pl
import torch
from utils import config as cfg
from utils.custom_classes.GarbageDataModule import GarbageDataModule
from utils.custom_classes.GarbageClassifier import GarbageClassifier
from utils.custom_classes.LossCurveCallback import LossCurveCallback

# # ========================
# # Training
# # ========================
# if __name__ == "__main__":
#     data_module = GarbageDataModule(batch_size=32)
#     data_module.setup()
    
#     model = GarbageClassifier(num_classes=data_module.num_classes, lr=1e-3)

#     trainer = pl.Trainer(max_epochs=3, accelerator="auto", devices=1)
#     trainer.fit(model, datamodule=data_module)
    
#     trainer.save_checkpoint(cfg.MODEL_PATH)
#     print(f"Model saved at {cfg.MODEL_PATH}")


# ========================
# Training
# ========================
if __name__ == "__main__":
    data_module = GarbageDataModule(batch_size=32)
    data_module.setup()
    
    model = GarbageClassifier(num_classes=data_module.num_classes, lr=1e-3)

    loss_curve_callback = LossCurveCallback(save_dir=cfg.LOSS_CURVES_PATH)

    trainer = pl.Trainer(
        max_epochs=3, 
        accelerator="auto", 
        devices=1,
        callbacks=[loss_curve_callback],
        num_sanity_val_steps=0
    )
    trainer.fit(model, datamodule=data_module)
    
    trainer.save_checkpoint(cfg.MODEL_PATH)
    print(f"Model saved at {cfg.MODEL_PATH}")
