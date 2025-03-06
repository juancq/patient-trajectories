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
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
#from lightning.pytorch.plugins import MixedPrecision


from halo.halo_data_module import HALODataModule
from halo.halo_improved import HALOModel
#from halo.model import HALOModel
#from halo.halo_rewrite import HALOModel
#from halo.halo_mixed_precision import HALOModel
#from halo.halo_logits import HALOModel
#from halo.halo_mixed_precision_coarse import HALOCoarseModel
from halo.config import HALOConfig
#from halo.config_64 import HALOConfig

def main():
    SEED = 4
    seed_everything(SEED)
    torch.set_float32_matmul_precision('high')

    dv = pl.read_excel("df_vars.xlsx")
    config = HALOConfig(dv)

    # Initialize data module
    data_path = Path('/mnt/data_volume/apdc/study1/preprocessed')
    #data_path = Path('/mnt/fast_drive')
    data_module = HALODataModule(config, batch_size=config.batch_size, 
                    train_name=data_path / 'df_train_halo.parquet',
                    val_name=data_path / 'df_tuning_halo.parquet',
                    #train_split=0.2,
                    )
    data_module.setup()
    #print(len(data_module.train_dataset))
    #print(len(data_module.val_dataset))
    #return
    dataset_size = len(data_module.train_dataset)
    steps_per_epoch = math.ceil(dataset_size / config.batch_size)
    total_steps = steps_per_epoch * config.epoch
    config.total_steps = total_steps
    config.warmup_steps = math.ceil(total_steps * 0.10)

    # Initialize model
    print(config)
    #model = HALOCoarseModel(config)
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
        filename='halo_model-{epoch:02d}-{val_loss:.4f}-{step}-n48_scheduler',
        save_top_k=2,
        mode='min',
    )

    checkpoint_path = None
    #checkpoint_path = 'save/halo_model-epoch=04-val_loss=0.000_128_flash_swig.ckpt'
    #checkpoint_path = 'save/halo_model-epoch=04-val_loss=0.000.ckpt'
    #checkpoint_path = 'save/halo_model-epoch=05-val_loss=0.000-step=199870.ckpt'
    #checkpoint_path = 'save/halo_model-epoch=16-val_loss=0.000-step=635103-n48.ckpt'
    #checkpoint_path = 'save/halo_model-epoch=09-val_loss=0.000-step=373590-n48_12x12.ckpt'
    #checkpoint_path = 'save/halo_model-epoch=49-val_loss=0.000-step=1867950-n48.ckpt'


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
        gradient_clip_val=1,
        #val_check_interval=0.35,
        #precision="bf16-mixed"
        #precision=MixedPrecision(precision='bf16-mixed',device='cuda', scaler=torch.amp.GradScaler())
        #profiler='advanced'
        #strategy=DDPStrategy(num_processes=8)
    )

    # Train the model
    if checkpoint_path:
        # resume
        trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    else:
        # from scratch
        trainer.fit(model, data_module)

    # Evaluate the model
    #trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    main()
