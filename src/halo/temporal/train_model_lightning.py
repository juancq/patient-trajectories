import os
import math

import torch
import numpy as np
import polars as pl
import random
import pickle
from pathlib import Path
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
#from lightning.pytorch.loggers import TensorBoardLogger
from halo_data_module import HALODataModule
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger


from halo import HALOModel
from config import HALOConfig

def main():
    SEED = 4
    seed_everything(SEED)
    torch.set_float32_matmul_precision('high')

    dv = pl.read_excel("temporal_df_vars.xlsx")
    config = HALOConfig(dv)

    # Initialize data module
    data_path = Path('/mnt/data_volume/apdc/study1/preprocessed')
    train_file = data_path / 'temporal_df_train_halo.parquet'
    val_file = data_path / 'temporal_df_train_halo.parquet'
    data_module = HALODataModule(config, 
                    batch_size=config.batch_size, 
                    train_name=train_file,
                    val_name=val_file,
    )
    data_module.setup()
    dataset_size = len(data_module.train_dataset)
    steps_per_epoch = math.ceil(dataset_size / config.batch_size)
    total_steps = steps_per_epoch * config.epoch
    config.total_steps = total_steps
    config.warmup_steps = math.ceil(total_steps * 0.10)

    # Initialize model
    print(config)
    model = HALOModel(config)
    model.transformer = torch.compile(model.transformer)
    model.ehr_head = torch.compile(model.ehr_head)#, mode='reduce-overhead')

    #mlf = MLFlowLogger(
    #    experiment_name='halo_study1',
    #    tracking_uri='file:./mlruns'
    #)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./save/',
        filename='halo_model-faster-lr5e3{epoch:02d}-{val_loss:.4f}-{step}',
        save_top_k=2,
        mode='min',
    )

    #checkpoint_path = 'save/halo_model-faster-epoch=15-val_loss=0.0001-step=487616.ckpt'
    #checkpoint_path = 'save/halo_model-faster-epoch=30-val_loss=0.0001-step=944756.ckpt'
    #checkpoint_path = 'save/halo_model-faster-epoch=32-val_loss=0.0001-step=1005708.ckpt'
    #checkpoint_path = 'save/halo_model-faster-epoch=36-val_loss=0.0001-step=1127612.ckpt'
    #checkpoint_path = None
    #checkpoint_path = 'save/halo_model-faster-lr5e3epoch=14-val_loss=0.0001-step=228570.ckpt'
    checkpoint_path = 'save/halo_model-faster-lr5e3epoch=18-val_loss=0.0001-step=289522.ckpt'

    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,  # number of epochs with no improvement after which training will be stopped
        mode='min'
    )

    # Trainer
    trainer = Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stopping_callback,
                RichProgressBar(leave=True)],
        # add mlflow here
        logger=True,
    )

    # Train the model
    if checkpoint_path:
        # resume
        trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    else:
        # from scratch
        trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
