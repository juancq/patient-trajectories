import sys
import argparse
import sklearn
import pandas as pd
import polars as pl
import numpy as np
from collections import defaultdict
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt


def plot_analysis(decile_stats, fname):
    plt.figure(figsize=(12,6))
    plt.plot(decile_stats['decile'], decile_stats['true_mean'], label='True Mean', marker='o')
    plt.plot(decile_stats['decile'], decile_stats['pred_mean'], label='Predicted Mean', marker='o')
    plt.xlabel('Decile')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(fname, bbox_inches='tight')


def decile_analysis(y_true, y_pred):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['decile'] = pd.qcut(df['y_true'], q=10, labels=False, duplicates='drop')
    decile_stats = df.groupby('decile').agg({
        'y_true': ['mean', 'count'],
        'y_pred': 'mean',
    }).reset_index()

    decile_stats.columns = ['decile', 'true_mean', 'count', 'pred_mean']
    decile_stats['error'] = decile_stats['pred_mean'] - decile_stats['true_mean']
    decile_stats['abs_error'] = abs(decile_stats['error'])
    decile_stats['rel_error'] = decile_stats['error'] / decile_stats['true_mean']
    return decile_stats


def get_embeddings(fname):
    logger.info(f'Reading {fname}')
    # reads embeddings
    df_embed = pl.scan_parquet(fname)
    df_embed = df_embed.with_columns(pl.col('ppn_int').cast(pl.Int32))
    n_unique = df_embed.select(pl.col('ppn_int').n_unique()).collect().item()
    len_ = df_embed.select(pl.len()).collect().item()
    logger.info(f'Embeddings unique ids and len: {n_unique}, {len_}')
    return df_embed

def main():
    """
    Fit a model on dataset, where features consist of embeddings and label
    is time to death. 
    """
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(20)
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    parent_path = Path('/mnt/data_volume/apdc/')
    label_path = parent_path / 'study1/preprocessed/'

    # have to use rbdm, since cod_urf is only available through the end of 2020
    death = pl.scan_parquet(parent_path / 'rbdm_processed.parquet')
    death = death.select(['ppn_int', 'DEATH_DATE'])

    for window in ['six_months', 'one_year', 'two_year']:
        logger.info(f'Window {window}')
        print(f'Window {window}')
        for task in ['length_of_stay', 'episode_count', 'cost']:

            label_file = label_path / f'tasks/{task}_{window}.parquet'
            #label_file = label_path / f'{task}_{window}.parquet'
            df = pl.scan_parquet(label_file)

            df = df.join(death, on='ppn_int', how='left', coalesce=True)
            df = df.with_columns(pl.col('DEATH_DATE').is_not_null().alias('died'))


            #df = df.collect(streaming=True).sample(n=TEST_SIZE, seed=123)
            df = df.collect(streaming=True)

            logger.info(f'Task: {task}')
            print(f'Task: {task}')
            # for storing all metrics for this window, gets saved as csv

            # outcome counts debugging
            if task in ['length_of_stay', 'episode_count']:
                t = df.select(pl.col(task).value_counts(sort=True)).unnest(task)
                t = t.with_columns(percent=pl.col('count')/df.shape[0])
                print(t)

            print(df.select(pl.col(task)).describe(
                percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
                )
            )

            print('< 0', df.select((pl.col(task) < 0).sum()))
            print('> 0', df.select((pl.col(task) > 0).sum()))
            print('==0', df.select((pl.col(task) == 0).sum()))


            print('-'*20)
            print('those who died')
            dead = df.filter(pl.col('died'))
            # outcome counts debugging
            if task in ['length_of_stay', 'episode_count']:
                t = dead.select(pl.col(task).value_counts(sort=True)).unnest(task)
                t = t.with_columns(percent=pl.col('count')/df.shape[0])
                print(t)

            print(dead.select(pl.col(task)).describe(
                percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
                )
            )

            print('< 0', dead.select((pl.col(task) < 0).sum()))
            print('> 0', dead.select((pl.col(task) > 0).sum()))
            print('==0', dead.select((pl.col(task) == 0).sum()))


if __name__ == "__main__":
    main()