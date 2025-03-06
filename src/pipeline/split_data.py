import sys
import numpy as np
import polars as pl
import polars.selectors as cs
import pickle
from tqdm import tqdm
from pathlib import Path
import duckdb

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(20)


def main():
    apdc_eddc_path = Path('/mnt/data_volume/apdc/study1')

    project_path = apdc_eddc_path / 'preprocessed' 

    # polars is easier to read, but duckdb is more memory efficient by an order of magnitude
    duckdb.query("SET enable_progress_bar=true;")
    for split in ['held_out']:
        query  = f"""
        SELECT * from read_parquet('{project_path}/full_cohort.parquet') as cohort
        join
            read_parquet('{split}_ppn.parquet') as ppn
        on 
            cohort.ppn_int=ppn.subject_id
        """
        duckdb.query(f'COPY ({query}) TO "{project_path}/full_cohort_{split}.parquet" (format parquet)')



    return
    # split full cohort

    df = pl.scan_parquet(project_path / 'full_cohort.parquet')
    print('df', df.select(pl.len()).collect())

    #for split in ['train', 'tuning', 'held_out']:
    for split in ['tuning', 'held_out']:
        ppn = pl.scan_parquet(f'{split}_ppn.parquet')
        ppn = ppn.with_columns(pl.col('subject_id').cast(pl.Int32).alias('ppn_int'))
        df_split = ppn.join(df, on='ppn_int', how='left', coalesce=True)
        df_split.collect(streaming=True).write_parquet(project_path / f'full_cohort_{split}.parquet')

    return
    # halo split
    df = pl.scan_parquet(project_path / 'df_ppn.parquet')
    print('df', df.select(pl.len()).collect())

    for split in ['train', 'tuning', 'held_out']:
        ppn = pl.scan_parquet(f'{split}_ppn.parquet')
        #print(len(ppn.collect(streaming=True)))
        ppn = ppn.rename({'subject_id': 'ppn_int'})
        #print(ppn.head().collect())
        #print(df.dtypes)
        df_split = df.join(ppn.select(pl.col('ppn_int').cast(pl.Int32)), on='ppn_int', coalesce=True)
        print('joined dataframe', df_split.select(pl.len()).collect(streaming=True))
        #print(df_split.head().collect())
        #continue
        df_split.collect(streaming=True).write_parquet(project_path / f'df_{split}_halo.parquet')


if __name__ == '__main__':
    main()