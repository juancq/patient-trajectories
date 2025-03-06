import numpy as np
from sklearnex import patch_sklearn

patch_sklearn()

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl
import polars.selectors as cs
import sklearn
import pacmap
import yaml
from box import Box
from loguru import logger

import utils.predict_utils as ut

streaming_flag = False


def conditional_set_memory_limit(value):
    def wrapper(func):
        if os.name == 'nt':
            from kill import set_memory_limit
            return set_memory_limit(value)(func)
        return func
    return wrapper


def get_embeddings(fname: str) -> pl.DataFrame:
    """
    Reads a parquet file of embeddings

    Args:
        fname (str): The file path to the parquet file

    Returns:
        pl.DataFrame: A polars DataFrame with the embeddings. The DataFrame
        has a column named 'ppn_int' which is cast to an integer type.
    """
    if not os.path.isfile(fname):
        raise ValueError(f'Invalid embedding path: {fname}')

    logger.info(f'Reading {fname}')
    # reads embeddings
    df_embed = pl.scan_parquet(fname)
    df_embed = df_embed.with_columns(pl.col('ppn_int').cast(pl.Int32))
    n_unique = df_embed.select(pl.col('ppn_int').n_unique()).collect().item()
    len_ = df_embed.select(pl.len()).collect().item()
    logger.info(f'Embeddings unique ids and len: {n_unique}, {len_}')
    return df_embed


def process_embeddings(embed_path, window, label_ids=None):
    embedding_set = {}
    for embed_name, path_to_embedding in embed_path.items():
        fname = Path(str(path_to_embedding).format(window=window))
        embedding = get_embeddings(fname)
        if label_ids: 
            embedding = label_ids.join(embedding, on='ppn_int', coalesce=True)
        embedding = embedding.sort('ppn_int')
        logger.info(f'Collecting {embed_name}')
        embedding = embedding.collect(streaming=streaming_flag)
        # stats
        n_unique = embedding.select(pl.col('ppn_int').n_unique()).item()
        len_ = embedding.select(pl.len()).item()
        logger.info(f'Embeddings after joining with label unique ids and len: {n_unique}, {len_}')

        embedding_set[embed_name] = embedding
    return embedding_set


def draw_scatter_plot(result, hue_data, color_palette, 
        plot_title, legend_title, legend_handles, **kwargs):
    plt.figure(figsize=(16,10))
    ax = sns.scatterplot(
        x=result[:,0], y=result[:,1],
        hue=hue_data, 
        # tab20
        palette=color_palette,
        legend="full",
        **kwargs
    )
    ax.set_title(plot_title)
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_alpha(1)
        #handle.set_sizes([100])
    legend_args = {'title':legend_title, 'markerscale':2}
    if legend_handles:
        ax.legend(
            handles, 
            legend_handles,
            **legend_args,
        )
    else:
        ax.legend(**legend_args)


@conditional_set_memory_limit(26)
def main():
    """
    Fit a model on dataset, where features consist of embeddings and label
    is time to death. 
    """
    main_start_time = time.time()
    pl.Config.set_tbl_cols(20)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    with open('config/predict_config.yaml', 'r') as file:
        yaml_config = yaml.safe_load(file)
    config = Box(yaml_config)

    w_temp = 'two_year'
    if os.name == 'nt':
        label_path = Path(config.label_path_nt)
        embed_path = {
            'esgpt':label_path / 'held_out_study1_two_year_embeddings_mean.pt',
            'halo': label_path / 'two_year_halo_embeddings_last_original_epoch34.parquet',
        }
        #embed_path = {
        #                'halo_original': label_path / f'{w_temp}_halo_embeddings_last_original.parquet',
        #                'halo_old': label_path / f'{w_temp}_halo_embeddings_last_old.parquet',
        #                #'halo_n48_epoch16': label_path / f'{w_temp}_halo_embeddings_last_n48_epoch16.parquet',
        #                #'halo_original_epoch0': label_path / f'{w_temp}_halo_embeddings_last_original_epoch0.parquet',
        #            }
    else:
        label_path = Path(config.label_path_linux)
        embed_path = {'esgpt':'/mnt/fast_drive/experiment/study1/pretrain/2024-08-14_14-11-29/embeddings/tuning_study1_{window}_embeddings_mean.pt',
                        #'halo': label_path / f'{w_temp}_halo_embeddings_last_v1.parquet',
                        #'halo_n48': label_path / f'{w_temp}_halo_embeddings_last_n48_v24.parquet',
                        #'halo_n48_v29': label_path / f'{w_temp}_halo_embeddings_last_v29.parquet',
                        'halo': label_path / 'two_year_halo_embeddings_last_original_epoch34.parquet',
                        #'esgpt': label_path / '{window}_halo_embeddings_avg_manual.parquet',
                        #'llm': label_path / '{window}_llm_embeddings.parquet',
                   }

    baseline_features = pl.scan_parquet(label_path / f'baseline_features_strong_two_year.parquet')
    baseline_features = baseline_features.fill_nan(0)
    baseline_features = baseline_features.fill_null(0)
    n_unique = baseline_features.select(pl.col('ppn_int').n_unique()).collect().item()
    len_ = baseline_features.select(pl.len()).collect().item()
    logger.info(f'Baseline features unique ids and len: {n_unique}, {len_}')

    ids = baseline_features.select(pl.col('ppn_int'))

    print(config.test_size)
    window = 'two_year'
    if config.test_size is not None: 
        test_size = int(config.test_size)
        if test_size <=0:
            raise ValueError(f'Test size must be a positive number: {test_size}')
        # for exploration, use subset to run fast
        ids = ids.collect(streaming=streaming_flag).sample(n=test_size, seed=config.seed)
        logger.info(f'Baseline features using ids {ids.select(pl.len()).item()}')
        ids = ids.lazy()

    # get the embeddings of only the ppns in label dataframe
    embedding_set = process_embeddings(embed_path, window)

    # baseline features
    df_baseline_window = embedding_set['halo'].join(baseline_features.collect(), on='ppn_int', coalesce=True)
    #embeddings = df_baseline_window.collect(streaming=streaming_flag)
    embeddings = df_baseline_window
    N = 500000
    embeddings = embeddings.head(N)
    #embeddings = embeddings.sample(50000, seed=123)

    print('^'*20)

    subset_data = ut.build_subsets(embeddings)

    X = embeddings.with_row_index()
    vectors = X.select(cs.matches(r"feature_\d"))
    features = X.select(~cs.matches(r"feature_\d"))
    features = features.rename(lambda c: c.replace('feature_', ''))
    ages = features.select((pl.col('age_recode')/10).floor()*10).get_column('age_recode')
    age_groups = ages.unique().sort().to_list()
    age_groups = [f'{c:.0f}' for c in age_groups] 
    #y = embedding_set['baseline'].get_column(task)

    mapper = pacmap.PaCMAP(n_components=2,
                            n_neighbors=None,
                            MN_ratio=0.5, FP_ratio=2.0, random_state=1)
    #result = mapper.fit_transform(pca_result)
    logger.info('Fitting pacmap')
    result = mapper.fit_transform(vectors)
    #tsne = TSNE(n_components=2, verbose=1)
    #tsne_result = tsne.fit_transform(pca_result)

    output_fname_suffix = f'_n_{N}'

    kwargs = {'s': 12, 'alpha':0.7}

    logger.info('Generating scatter plot for ages')
    draw_scatter_plot(result, hue_data=ages, 
        color_palette=sns.color_palette('viridis', n_colors=len(age_groups)),
        plot_title='Patient Trajectory Embeddings',
        legend_title='Ages',
        legend_handles=age_groups,
        **kwargs,
    )
    plt.savefig(f"clustering_age_pacmap_yeo2{output_fname_suffix}.png", bbox_inches='tight', dpi=300) 

    # for sex plot
    sex = subset_data['female']
    label = ['male', 'female']
    logger.info('Generating scatter plot for sex')
    draw_scatter_plot(result, hue_data=sex,
        color_palette=sns.color_palette('Set1', n_colors=len(label)),
        plot_title='Patient Trajectory Embeddings',
        legend_title='Sex',
        legend_handles=label,
        **kwargs,
    )
    plt.savefig(f"clustering_sex_pacmap{output_fname_suffix}.png", bbox_inches='tight', dpi=300) 

    # for age-sex plot
    #logger.info('Generating scatter plot for age-sex')
    #plt.figure(figsize=(16,10))
    #ax = sns.scatterplot(
    #    x=result[:,0], y=result[:,1],
    #    hue=ages, 
    #    style=sex, 
    #    palette=sns.color_palette("tab20", len(age_groups)),
    #    legend="full",
    #    alpha=0.3
    #)
    #ax.set_title("Plot of Clinical History Embeddings")
    #handles, labels = ax.get_legend_handles_labels()
    ##ax.legend(handles, label, title="Sex")
    #plt.savefig(f"clustering_age_sex_pacmap{output_fname_suffix}.png", bbox_inches='tight', dpi=300) 


    # for number of episodes
    episodes = features.select(
        pl.when(pl.col('num_episodes')==1).then(pl.lit('=1'))
        .when(pl.col('num_episodes').is_between(2,4)).then(pl.lit('2-4'))
        .otherwise(pl.lit('=5'))
        .alias('episode_group')
    ).get_column('episode_group')

    #episodes = features.select(
    #    pl.when(pl.col('num_episodes')==1).then(0)
    #    .when(pl.col('num_episodes').is_between(2,4)).then(1)
    #    .otherwise(2)
    #    .alias('episode_group')
    #).get_column('episode_group')
    #label = ['episodes=1', 'episodes=2-4', 'episodes>=5']
    logger.info('Generating scatter plot for episode counts')
    #plt.figure(figsize=(16,10))
    #ax = sns.scatterplot(
    #    x=result[:,0], y=result[:,1],
    #    hue=episodes, 
    #    palette=['#E41A1C', '#377EB8', '#4DAF4A'],
    #    legend="full",
    #    alpha=0.4
    #)
    draw_scatter_plot(result, hue_data=episodes, 
        color_palette=sns.color_palette('Set1', n_colors=3),
        plot_title='Patient Trajectory Embeddings',
        legend_title='Episode Count',
        legend_handles=None,
        **kwargs,
    )
    plt.savefig(f"clustering_episode_pacmap{output_fname_suffix}.png", bbox_inches='tight', dpi=300) 


    diagnoses = {
        'cvd': 'I[1-7]',
        'diabetes': 'E1[014]',
        'ckd': 'N1[89]|E[1-4].2|I1[23]|I15.[01]|N1[124-6]|N2[5-8]|N39[12]|D59.3|B52.0|E85.3',
        'mental_health': 'F[2-4]|X[67]|X8[0-4]',
        'cancer': 'C[0-9]|D4[56],D47.[13-5]',
        'trauma': 'S|T[0-6]|T7[0-5,9]',
        'birth': 'O8[0-4]',
    }
    # for diagnosis subgroups
    diagnosis_groups = features.select(
        pl.when((pl.col('diagnosis_history').list.eval(pl.element().str.contains(diagnoses['cvd']))).list.any())
        .then(pl.lit('cvd'))
        .when(
            (pl.col('diagnosis_history').list.eval(pl.element().str.contains(diagnoses['diabetes'])))
            .list.any()
            )
        .then(pl.lit('diabetes'))
        .when(
            (pl.col('diagnosis_history').list.eval(pl.element().str.contains(diagnoses['ckd'])))
            .list.any()
            )
        .then(pl.lit('ckd'))
        .when(
            (pl.col('diagnosis_history').list.eval(pl.element().str.contains(diagnoses['mental_health'])))
            .list.any()
            )
        .then(pl.lit('mental_health'))
        .when(
            (pl.col('diagnosis_history').list.eval(pl.element().str.contains(diagnoses['cancer'])))
            .list.any()
            )
        .then(pl.lit('cancer'))
        .when(
            (pl.col('diagnosis_history').list.eval(pl.element().str.contains(diagnoses['trauma'])))
            .list.any()
            )
        .then(pl.lit('trauma'))
        .when(
            (pl.col('diagnosis_history').list.eval(pl.element().str.contains(diagnoses['birth'])))
            .list.any()
            )
        .then(pl.lit('birth'))
        .otherwise(pl.lit('other'))
        .alias('diagnosis_group')
    )#.get_column('diagnosis_group')

    # get those with 'other' diagnosis
    mask = diagnosis_groups.select((pl.col('diagnosis_group')=='other').cast(pl.Boolean)).to_numpy().flatten()
    diagnosis_groups = diagnosis_groups.filter(~(pl.col('diagnosis_group')=='other'))
    result = result[~mask]
    diagnosis_groups = diagnosis_groups.get_column('diagnosis_group')

    logger.info('Generating scatter plot for diagnoses')
    draw_scatter_plot(result, 
        hue_data=diagnosis_groups, 
        color_palette=sns.color_palette("Paired", n_colors=len(diagnoses)+1),
        plot_title='Patient Trajectory Embeddings',
        legend_title='Diagnosis',
        legend_handles=None,
        **kwargs,
    )
    plt.savefig(f"clustering_diagnosis_pacmap{output_fname_suffix}.png", bbox_inches='tight', dpi=300) 

    main_end_time = time.time()
    runtime = main_end_time - main_start_time
    print(f'Execution time of script: {runtime/60:.1f} minutes')


if __name__ == "__main__":
    main()
