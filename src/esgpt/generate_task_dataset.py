#!/usr/bin/env python
"""Builds a dataset given a hydra config file."""

import numpy as np
import polars as pl
import hydra
from pathlib import Path
from omegaconf import DictConfig
from loguru import logger

from EventStream.data.dataset_polars import Dataset

streaming_flag = False


@hydra.main(version_base=None)
def main(cfg: DictConfig):

    dataset_dir = Path(cfg.dataset_dir)

    #Dataset._PICKLER = 'pickle'
    #ESD = Dataset.load(dataset_dir)
    #events = ESD.events_df.lazy()
    #events = ESD.tuning_events_df.lazy()

    split = 'held_out'

    task_df_dir = dataset_dir / "task_dfs" 
    task_df_dir.mkdir(exist_ok=True, parents=False)
    task = 'study1'

    # load prediction point 
    path = Path('/mnt/data_volume/apdc/study1/preprocessed/tasks')
    # can use any of the task files, because the index date is the same across all
    # (this means index date should be stored separately)
    #windows = ['six_months', 'one_year', 'two_year']
    windows = ['two_year']
    logger.info(f'Processing split {split}')
    for window in windows:
        logger.info(f'Window {window}')
        index_date_fname = path / f'length_of_stay_{window}.parquet'
        #index_date_fname = path / f'episode_count_{window}.parquet'
        index_date = pl.scan_parquet(index_date_fname)
        index_date = index_date.with_columns(pl.col('ppn_int').cast(pl.UInt32))
        #redux = pl.scan_csv('hello.csv').select(pl.col('ppn_int').cast(pl.UInt32))
        #redux = redux.unique('ppn_int')
        #index_date = index_date.join(redux, on='ppn_int')
        logger.info(f'Using index date from {index_date_fname}')
        # assumes ppn is unsigned integer, which should always be the case if 
        # conversion from string ppn to int ppn just takes the number from the string
        print(index_date.head().collect())
        #index_date = index_date.rename({'ppn_int': 'subject_id',
        #                                'index_datetime': 'end_time',
        #                                'length_of_stay': 'label'})
        index_date = index_date.select(
            pl.col('ppn_int').alias('subject_id'),
            pl.col('index_datetime').alias('end_time'),
            pl.col('length_of_stay').alias('label')
        )

        print(index_date.head().collect())
        index_date = index_date.with_columns(
                    pl.lit(None, dtype=pl.Datetime).alias('start_time'), 
                    # adding some delta because end_time is open interval
                    pl.col('end_time').cast(pl.Datetime)+pl.duration(milliseconds=100)
        ) 
        print(index_date.head().collect())

        index_date = index_date.collect(streaming=streaming_flag)
        task_dataset_name = task_df_dir / f"{task}_{window}.parquet"
        logger.info(f'Saving {task_dataset_name}')
        index_date.write_parquet(task_dataset_name, use_pyarrow=True)


if __name__ == "__main__":
    main()
