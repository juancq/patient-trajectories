#!/usr/bin/env python
"""Builds a dataset given a hydra config file."""

import numpy as np
import polars as pl
import hydra
from pathlib import Path
from omegaconf import DictConfig

from EventStream.data.dataset_polars import Dataset


@hydra.main(version_base=None)
def main(cfg: DictConfig):

    dataset_dir = Path(cfg.dataset_dir)

    Dataset._PICKLER = 'pickle'
    ESD = Dataset.load(dataset_dir)
    events = ESD.events_df.lazy()

    #print(events.head().collect())
    #print(events.filter(pl.col('subject_id')==1210).collect())

	## get last year data
    #events = events.sort('subject_id')
    #events = events.select(pl.col('subject_id'),(pl.col('timestamp').max()-pl.duration(days=365)).over('subject_id'))
    #events = events.group_by('subject_id').last()
    #print(events.collect())
    #return

    # calculate date of last event of type admission
    events = events.with_columns(
        pl.col('timestamp').filter(pl.col('event_type').cast(pl.Utf8).str.contains('ADMISSION')).max()
        .over('subject_id').alias('last_admission'))
    # keep the n-1 records by starting from dates before last admission
    events = events.filter(pl.col('timestamp') < pl.col('last_admission'))
    events = events.select(pl.exclude("last_admission"))
    events = events.group_by('subject_id').last()
    #print(events.filter(pl.col('subject_id')==1210).collect())
    #return

    task_df_dir = dataset_dir / "task_dfs" 
    task_df_dir.mkdir(exist_ok=True, parents=False)
    height = events.select(pl.col('subject_id')).collect().height

    label = pl.Series('label', np.random.choice([True, False], size=height))
    events = events.with_columns(
                pl.lit(None, dtype=pl.Datetime).alias('start_time'), 
                pl.col('timestamp').alias('end_time')+pl.duration(minutes=1), # adding a minute because end_time is open interval
                label
                )
    events = events.select(pl.col(['subject_id', 'start_time', 'end_time', 'label']))
    task = 'time_to_death'
    #task = 'cost'
    events.collect().write_parquet(task_df_dir / f"{task}.parquet")

if __name__ == "__main__":
    main()
