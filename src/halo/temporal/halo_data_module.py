import os
from functools import partial
import torch
import numpy as np
import polars as pl
import random
import pickle
import lightning as L
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from numba import jit

streaming_flag = True

def collate_fn(batch, mask_shift):
    if len(batch[0]) == 2:
        batch_ehr, batch_mask = zip(*batch)
    else:
        batch_ehr, batch_mask, batch_ppn = zip(*batch)
    batch_ehr = np.concatenate(batch_ehr)
    batch_mask = np.concatenate(batch_mask)
    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32)
    batch_mask = torch.tensor(batch_mask, dtype=torch.float32)
    batch_mask[:, 1] = 1  # Set the mask to cover the labels
    if mask_shift:
        batch_mask = batch_mask[:, 1:, :]  # Shift mask to match the shifted labels and predictions the model will return
    if len(batch[0]) == 2:
        return batch_ehr, batch_mask
    else:
        return batch_ehr, batch_mask, batch_ppn

class EHRDataset(Dataset):
    def __init__(self, ehr_data, config, include_id=False):
        self.ehr_data = ehr_data
        self.config = config
        self.include_id = include_id
    
    def __len__(self):
        return len(self.ehr_data)
    
    def __getitem__(self, idx):
        # patient's entire data is contained within a single row
        patient = self.ehr_data.row(idx, named=True)
        # (1, number of visits, vocab_size)
        ehr = np.zeros((1,self.config.n_ctx, self.config.total_vocab_size))
        # (1, number of visits, 1)
        mask = np.zeros((1, self.config.n_ctx, 1))
        # Set the first visits to be the start token
        ehr[0,0, self.config.start_token_idx] = 1  
        # labels go on the second position
        ehr[0,1][patient['labels']] = 1  # Set the patient labels
        max_num_visits = min(len(patient["visits"]), self.config.n_ctx - 2)
        for visit_idx, visit in enumerate(patient["visits"]):
            if visit_idx >= max_num_visits:  # leave space for start token and labels
                break
            ehr[0,visit_idx + 2][visit] = 1
            mask[0,visit_idx + 2] = 1
        ehr[0, max_num_visits + 1, self.config.end_token_idx] = 1  # Set final visit to have the end token
        ehr[0, max_num_visits + 2:, self.config.pad_token_idx] = 1  # Set the rest to the padded visit token

        if self.include_id:
            return ehr, mask, patient['ppn_int']
        return ehr, mask


class HALODataModule(L.LightningDataModule):
    def __init__(self, config, batch_size, train_name, 
            val_name=None, train_split=1, shuffle=True):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.train_name = train_name
        self.val_name = val_name
        self.train_split = train_split
        self.shuffle = shuffle
        self.train_dataset = None
        self.val_dataset = None
        self.mask_shift = True

    def setup(self, stage=None):
        if stage == 'embeddings':
            include_id = True
            self.shuffle = False
            self.mask_shift = False
        else:
            include_id = False

        df_trn = pl.scan_parquet(self.train_name)
        train_ehr_dataset = df_trn.collect(streaming=streaming_flag)
        if self.train_split < 1:
            train_ehr_dataset = train_ehr_dataset.sample(fraction=self.train_split, seed=1)
        self.train_dataset = EHRDataset(train_ehr_dataset, self.config, include_id=include_id)

        if self.val_name:
            val_ehr_dataset = pl.read_parquet(self.val_name)
            if self.train_split < 1:
                val_ehr_dataset = val_ehr_dataset.sample(fraction=self.train_split, seed=1)
            self.val_dataset = EHRDataset(val_ehr_dataset, self.config, include_id=include_id)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                shuffle=self.shuffle, num_workers=self.config.num_workers, 
                pin_memory=self.config.pin_memory,
                collate_fn=partial(collate_fn, mask_shift=self.mask_shift)
                )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                pin_memory=self.config.pin_memory,
                shuffle=False, num_workers=self.config.num_workers, 
                collate_fn=partial(collate_fn, mask_shift=self.mask_shift)
        )
