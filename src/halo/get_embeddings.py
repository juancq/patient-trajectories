import argparse
import os
import pickle
import random
import sys
import torch
import yaml
import numpy as np
import polars as pl

from loguru import logger
from pathlib import Path
from tqdm import tqdm

from lightning.pytorch import Trainer, seed_everything

from halo.model import HALOModel
#from config_64 import HALOConfig
from halo.config import HALOConfig
from halo.halo_data_module import HALODataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pooling_method', type=str, default='last')
    args = parser.parse_args()
    SEED = 123
    seed_everything(SEED)
    torch.set_float32_matmul_precision('high')

    dv = pl.read_excel("halo/df_vars.xlsx")

    device = 'cuda'
    pooling_method = args.pooling_method

    #windows = ['six_months', 'one_year', 'two_year']
    windows = ['two_year']

    with open('config/best_models.yml', 'r') as file:
        model_list = yaml.safe_load(file)

    apdc_eddc_path = Path('/mnt/data_volume/apdc/study1')
    data_path = apdc_eddc_path / 'preprocessed' 
    # path to date for each patient up to an index date
    index_date_path = data_path / 'tasks'
    split = 'held_out'

    model_list = {'old':model_list['old']}
    for model_name, model_settings in model_list.items():
        config = HALOConfig(dv, **model_settings['model'])
        config.batch_size = 256
        checkpoint_path = model_settings['path']
        save_postfix = model_name
        logger.info(f'Generating embeddings for model {checkpoint_path}')

        # load model checkpoint
        model = HALOModel(config)
        #checkpoint_path = 'save/halo_model-epoch=20-val_loss=0.000_128_flash_swig.ckpt'
        #checkpoint_path = 'save/halo_model-epoch=49-val_loss=0.000-step=1867950-n48.ckpt'
        #checkpoint_path = 'save/halo_model-epoch=00-val_loss=0.000-step=37359-n48_12x12-v2.ckpt'

        checkpoint = torch.load(checkpoint_path, weights_only=True)
        state_dict = {k.replace('_orig_mod.', ''):v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        # compile
        model.transformer = torch.compile(model.transformer)
        model.ehr_head = torch.compile(model.ehr_head)
        model.to(device)

        for window in tqdm(windows):
            # Initialize data module
            data_module = HALODataModule(config, batch_size=config.batch_size,
                            train_name=index_date_path / f"df_{split}_halo_{window}_v2.parquet")
            data_module.setup(stage='embeddings')

            embeddings = []

            logger.info(f'Generating embeddings for {window}')
            model.eval()
            iteration = 0
            df = data_module.train_dataset.ehr_data
            dataloader = data_module.train_dataloader()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader)):
                    batch_ehr, batch_mask, batch_ppn = batch
                    batch_ehr = batch_ehr.to(device)
                    batch_mask = batch_mask.to(device)
                    hidden_states = model.transformer(batch_ehr, position_ids=None, past=None)
                    # mean pooling of embeddings - naive implementation
                    if pooling_method == 'naive_avg':
                        batch_embeddings = torch.mean(hidden_states, dim=1)
                        batch_embeddings = batch_embeddings.cpu().detach().numpy()
                        #print(batch_embeddings.shape)
                        #return
                    elif pooling_method == 'avg':
                        # masked mean
                        #print(hidden_states.shape, batch_mask.shape)
                        #masked = torch.mul(hidden_states, batch_mask)
                        masked = hidden_states * batch_mask
                        batch_embeddings = masked.sum(dim=1) / batch_mask.sum(dim=1)
                        batch_embeddings = batch_embeddings.cpu().detach().numpy()
                        #batch_embeddings.tofile("temp.csv", sep=',')
                        #print(batch_embeddings.shape)
                    elif pooling_method in ['avg_manual', 'last']:
                    
                        # temp
                        hidden_states = hidden_states.cpu().detach().numpy()
                        df_temp = df.slice(i*config.batch_size, config.batch_size)
                        # gets the actual number of visits for each patient in the batch
                        num_visits = (df_temp.with_columns(pl.col("visits").list.len().alias("len"))
                                        .select("len").to_numpy().squeeze())
                        # truncates visits to max seq len of model
                        num_visits = np.clip(num_visits, a_min=None, a_max=config.n_ctx - 2)

                        batch_embeddings = []
                        if pooling_method == 'avg_manual':
                            # for every patient in the batch, get the embedding for the last visit
                            for j in range(len(batch_ehr)):
                                # for patient j, get the embedding at their last visit, 
                                # :1 because first embedding is for the start visit special token
                                # +2 so that the indexing includes num_visits[j]+1 (+2 because first two spots are for start and labels)
                                pool = np.mean(hidden_states[j, 2:num_visits[j] + 2, :], axis=0)
                                # make it 2d by adding a dimension
                                pool = np.expand_dims(pool, axis=0)
                                batch_embeddings.append(pool)
                        elif pooling_method == 'last':
                            # for every patient in the batch, get the embedding for the last visit
                            for j in range(len(batch_ehr)):
                                # for patient j, get the embedding at their last visit, 
                                # +1 because first embedding is for the start visit special token and second embedding is for labels
                                last_visit_embedding = hidden_states[j, num_visits[j] + 1, :]
                                # make it 2d by adding a dimension
                                batch_embeddings.append(np.expand_dims(last_visit_embedding, axis=0))
                            
                        batch_embeddings = np.concatenate(batch_embeddings)
                    embeddings.append(batch_embeddings)

                    if i % 100 == 0:
                        np.save(f'__{iteration}__.npy', np.concatenate(embeddings))
                        iteration += 1
                        embeddings = []
                    
            if embeddings:
                np.save(f'__{iteration}__.npy', np.concatenate(embeddings))
                iteration += 1
                embeddings = []

            embeddings = [np.load(f'__{i}__.npy') for i in range(iteration)]
            embeddings = np.concatenate(embeddings)
            for i in range(iteration):
                os.remove(f'__{i}__.npy')
            df_embeddings = pl.DataFrame(embeddings, 
                        schema=[f'feature_{x}' for x in range(embeddings.shape[-1])])
            df_embeddings = df_embeddings.with_columns(df.get_column('ppn_int'))
            df_embeddings.write_parquet(index_date_path / f"{window}_halo_embeddings_{pooling_method}_{save_postfix}.parquet", 
                        use_pyarrow=True)


def _backup_main():
    SEED = 123
    seed_everything(SEED)
    torch.set_float32_matmul_precision('high')

    dv = pl.read_excel("df_vars.xlsx")

    config = HALOConfig(dv)
    config.batch_size = 256

    apdc_eddc_path = Path('/mnt/data_volume/apdc/study1')

    project_path = apdc_eddc_path / 'preprocessed' 
    split = 'tuning'
    df = pl.scan_parquet(project_path / f'full_cohort_{split}.parquet')

    # prediction point path for various tasks
    index_date_path = project_path / 'tasks'

    windows = ['six_months', 'one_year', 'two_year']
    for window in tqdm(windows):
        index_date = pl.scan_parquet(index_date_path / f'los_{window}.parquet')
        df_mapped.write_parquet(index_date_path / f"df_{split}_halo_{window}.parquet")
        df_window = df.join(index_date, on='ppn_int', coalesce=True)
        df_window = df_window.filter(
            (pl.col('episode_start_date') < pl.col('index_date'))
            .over('ppn_int')
        )

        df_mapped = map_data(df_window)

        print('Processing ...')
        df_mapped = df_mapped.collect(streaming=True)
        # statistics
        num_visits = df_mapped.select(pl.col('visits').list.len())
        num_patients = len(df_mapped)
        print(f"Number of patients: {num_patients}")
        print(f"Maximum number of visits: {max(num_visits)}")
        print(f"Average number of visits: {np.mean(np.array(num_visits))}")

        print('Writing parquet')
        df_mapped.write_parquet(index_date_path / f"df_{split}_halo_{window}.parquet")

    # Initialize data module
    data_path = Path('/mnt/data_volume/apdc/study1/preprocessed')
    data_module = HALODataModule(config, batch_size=config.batch_size,
                    train_name=data_path / 'df_tuning_halo.parquet',
                    )
    data_module.setup(stage='embeddings')

    model = HALOModel(config)
    checkpoint = torch.load('save/halo_model-epoch=20-val_loss=0.000_128_flash_swig.ckpt',
                        weights_only=True)
    state_dict = {k.replace('_orig_mod.', ''):v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.transformer = torch.compile(model.transformer)
    model.ehr_head = torch.compile(model.ehr_head)
    device = 'cuda'
    model.to(device)

    # Embed and cluster dataset
    hs_list = []
    cohort = 'test'

    logger.info(f'Generating embeddings for {cohort}')
    model.eval()
    iteration = 0
    df = data_module.val_dataset.ehr_data
    val_dataloader = data_module.val_dataloader()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            batch_ehr, batch_mask, batch_ppn = batch
            batch_ehr = batch_ehr.to(device)
            hidden_states = model.transformer(batch_ehr, position_ids=None, past=None)
            pooled_embeddings = torch.mean(hidden_states, dim=1)
            pooled_embeddings = pooled_embeddings.cpu().detach().numpy()
            hs_list.append(pooled_embeddings)

            #print(pooled_embeddings.shape)
            '''
            hidden_states = hidden_states.cpu().detach().numpy()
            # use this for last embedding
            df_temp = df.slice(i*config.batch_size, config.batch_size)
            # gets the actual number of visits for each patient in the batch
            num_visits = (df_temp.with_columns(pl.col("visits").list.len().alias("len"))
                            .select("len").to_numpy().squeeze())
            #import code
            #code.interact(local=locals())
            # truncates visits to max seq len of model
            num_visits = np.clip(num_visits, a_min=None, a_max=config.n_ctx - 2)

            # for every patient in the batch, get the embedding for the last visit
            for j in range(len(batch_ehr)):
                # for patient j, get the embedding at their last visit, 
                # +1 because first embedding is for the start visit special token
                # make it 2d by adding a dimension
                hs_list.append(np.expand_dims(hidden_states[j, num_visits[j] + 1, :], axis=0))
            '''

            #from IPython import embed
            #embed()
            #return
            if i % 100 == 0:
                np.save(f'__{iteration}__.npy', np.concatenate(hs_list))
                iteration += 1
                hs_list = []
            
    if hs_list:
        np.save(f'__{iteration}__.npy', np.concatenate(hs_list))
        iteration += 1
        hs_list = []

    hs_list = [np.load(f'__{i}__.npy') for i in range(iteration)]
    hs = np.concatenate(hs_list)
    for i in range(iteration):
        os.remove(f'__{i}__.npy')
    embeddings = pl.DataFrame(hs, schema=[f'feature_{x}' for x in range(hs.shape[-1])])
    embeddings = embeddings.with_columns(df.get_column('ppn_int'))
    embeddings.write_parquet(f"{cohort}_embeddings.parquet", use_pyarrow=True)
    r
                            
if __name__ == '__main__':
    main()
