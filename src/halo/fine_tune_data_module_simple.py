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

def collate_fn(batch, mask_shift):
    if len(batch[0]) == 2:
        batch_ehr, batch_mask = zip(*batch)
    else:
        batch_ehr, batch_mask, batch_label = zip(*batch)
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
        #return batch_ehr, batch_mask, batch_label
        return {
            'input_ids': batch_ehr, 
            'attention_mask': batch_mask, 
            'labels': torch.tensor(batch_label)
        }

class EHRDataset(Dataset):
    def __init__(self, ehr_data, config, task, include_id=False):
        self.ehr_data = ehr_data
        self.config = config
        self.include_id = include_id
        self.task = task
    
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
        return ehr, mask, patient[self.task]


class HALODataModule(L.LightningDataModule):
    def __init__(self, config, batch_size, 
            train_data, val_data, test_data, 
            task, shuffle=True):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.task = task
        self.mask_shift = False

    def setup(self, stage=None):
        if stage == 'embeddings':
            include_id = True
            self.shuffle = False
        else:
            include_id = False

        if self.train_data is not None:
            self.train_dataset = EHRDataset(self.train_data, self.config, self.task, include_id=include_id)

        if self.val_data is not None:
            self.val_dataset = EHRDataset(self.val_data, self.config, self.task, include_id=include_id)

        if self.test_data is not None:
            self.test_dataset = EHRDataset(self.test_data, self.config, self.task, include_id=include_id)

    def get_loader(self, split, subset_data=None):
        data = None
        if split == 'train':
            data = self.train_dataset
        elif split == 'val':
            data = self.val_dataset
        elif split == 'test':
            data = self.test_dataset
        elif split == 'subset' and subset_data is not None:
            data = subset_data
        else:
            raise Exception('Error, incorrect split', split)
        
        return DataLoader(data, batch_size=self.batch_size, 
                shuffle=self.shuffle, num_workers=self.config.num_workers, 
                pin_memory=self.config.pin_memory,
                collate_fn=partial(collate_fn, mask_shift=self.mask_shift)
                )

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('val')

    def test_dataloader(self):
        return self.get_loader('test')
