import os
import glob
import polars as pl
from pathlib import Path
from tqdm import tqdm
import time
import duckdb


def main2():
    #os.environ['POLARS_MAX_THREADS'] = "1"
    data_path = '/mnt/data_volume/apdc'

    for fname in tqdm(sorted(glob.glob(f'{data_path}/str_group_files/apdc_group*.parquet'))):
        cohort = pl.scan_parquet(fname)
        cohort_static = cohort.group_by('ppn_int').first()
        out_name = Path(fname).name
        out_name = out_name.replace('.parquet', '_static.parquet')
        cohort_static.collect().write_parquet(f'{data_path}/preprocessed/{out_name}')


def main():
    start= time.time()
    root_path = Path('/mnt/data_volume/apdc/study1/preprocessed')

    duckdb.execute("PRAGMA enable_progress_bar;")
    duckdb.execute("PRAGMA threads=6;")
    duckdb.execute("PRAGMA memory_limit='50GB';")

    out_name = 'apdc.parquet'
    query = f"""
    COPY (
        SELECT * FROM
            (
                SELECT *, 
                ROW_NUMBER() 
                OVER(PARTITION BY ppn_int) as rn FROM '{root_path}/group_*.parquet'
            ) sub
            WHERE rn = 1
    ) to '{root_path}/{out_name}' (FORMAT 'PARQUET', CODEC 'ZSTD');
        """
    duckdb.sql(query)
    return





    df = pl.scan_parquet(root_path / 'group_*.parquet')

    cohort_static = df.group_by('ppn_int').first()
    cohort_static.collect(streaming=True).write_parquet(root_path / 'esgpt_static.parquet')
    end = time.time()
    print('Time', end-start, (end-start)/60)

if __name__ == '__main__':
    main()
