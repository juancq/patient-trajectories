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

from halo import HALOModel
from config import HALOConfig
from halo_data_module import HALODataModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--pooling_method', type=str, default='last')
    parser.add_argument('--input_files', nargs='+', type=str, help='files for embedding generation')
    parser.add_argument('--output_path', type=str, help='path for outputs')
    args = parser.parse_args()

    output_path = Path(args.output_path)
    SEED = 123
    seed_everything(SEED)
    torch.set_float32_matmul_precision('high')

    if args.model is None:
        logger.info('Missing model argument')
        return

    dv = pl.read_excel("temporal_df_vars.xlsx")

    device = 'cuda'
    pooling_method = args.pooling_method

    with open('best_models.yml', 'r') as file:
        model_list = yaml.safe_load(file)

    apdc_eddc_path = Path('/mnt/data_volume/apdc/study1')
    data_path = apdc_eddc_path / 'preprocessed' 
    # path to date for each patient up to an index date
    index_date_path = data_path / 'tasks'

    model_name = args.model
    if model_name not in model_list:
        logger.error('Model name not found in model yaml. Add model name and architecture details to best_models.yaml')
    model_settings = model_list[model_name]

    config = HALOConfig(dv, **model_settings['model'])
    config.batch_size = 512
    checkpoint_path = model_settings['path']
    logger.info(f'Generating embeddings using model checkpoint {checkpoint_path}')

    # load model checkpoint
    model = HALOModel(config)

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    state_dict = {k.replace('_orig_mod.', ''):v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    # compile
    model.transformer = torch.compile(model.transformer)
    model.ehr_head = torch.compile(model.ehr_head)
    model.to(device)

    for input_file in args.input_files:
        input_file = Path(input_file)
        # Initialize data module
        data_module = HALODataModule(
            config, 
            batch_size=config.batch_size,
            train_name=input_file
        )
        data_module.setup(stage='embeddings')

        embeddings = []

        logger.info(f'Generating embeddings for {input_file.name}')
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
        input_file = input_file.with_stem(f'{input_file.stem}_embeddings')
        df_embeddings.write_parquet(
            output_path / input_file.name,
            use_pyarrow=True
        )

                            
if __name__ == '__main__':
    main()