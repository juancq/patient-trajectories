# for records
# group by firstadm, sort by episode_sequence_number
# take first one


# for procedures
# group by firstadm, sort by episode_sequence_number
# set episode_start_date of group to firstadm
# then melt everything as usual (all procedures will get same date)


import argparse
import duckdb
import polars as pl
from pathlib import Path
from loguru import logger

from melt_procedures import melt_apdc_procedures
from melt_procedures import melt_apdc_diagnosis

import os


def main():
    os.environ['POLARS_MAX_THREADS'] = "1"

    parser = argparse.ArgumentParser()
    parser.add_argument('--condense', help='rerun everything', default=False, action='store_true')
    parser.add_argument('--add_stay_number', help='add stay number', default=False, action='store_true')
    parser.add_argument('--melt_procedures', help='melt procedure codes', default=False, action='store_true')
    parser.add_argument('--melt_diagnosis', help='melt diagnosis codes', default=False, action='store_true')
    args = parser.parse_args()

    #data_path = Path('/mnt/data_volume/apdc')
    #name = 'five_year_predeath_apdc.parquet'

    data_path = Path('/mnt/data_volume/apdc/lumos')
    name = 'lumos_dataset.parquet'
    #out_name = name.replace('.parquet', '_condensed.parquet')
    #out_name = name.replace('.parquet', '_processed.parquet')
    out_name = name

    if args.condense: 
        logger.info(f'Condensing {name}')
        query = f"""
        COPY (
        SELECT * FROM
            (SELECT *, ROW_NUMBER() OVER(PARTITION BY ppn_int,firstadm 
            ORDER BY transseq) as rn
            FROM read_parquet('{data_path}/{name}')) sub
            WHERE rn = 1
        ) to '{data_path}/{out_name}' (FORMAT 'PARQUET', CODEC 'ZSTD');
            """

        query = f"""
        COPY (
        SELECT * FROM
            (SELECT *, ROW_NUMBER() over ppn_group as rn,
            max(case when nest > 0 then 1 else 0 end) over ppn_group as nested_group,
            sum(nest) over ppn_group as nested_sum
            FROM read_parquet('{data_path}/{name}')
            WINDOW 
                ppn_group AS (
                    PARTITION BY ppn_int,firstadm 
                    ORDER BY transseq) 
            ) sub
            WHERE rn = 1
        ) to '{data_path}/{out_name}' (FORMAT 'PARQUET', CODEC 'ZSTD');
            """
        # much faster
        duckdb.sql(query)

    if args.melt_procedures: 
        df = pl.scan_parquet(data_path / name)
        # set all episode start dates to first admission, to preserve procedures
        # even when dropping transfers
        df = df.with_columns(pl.col('firstadm').str.to_date('%m/%d/%Y').alias('episode_start_date'))
        melt_apdc_procedures(data_path, out_name, df)

    if args.melt_diagnosis: 
        df = pl.scan_parquet(data_path / name)
        # set all episode start dates to first admission, to preserve procedures
        # even when dropping transfers
        df = df.with_columns(pl.col('firstadm').str.to_date('%m/%d/%Y').alias('episode_start_date'))
        melt_apdc_diagnosis(data_path, out_name, df)


    if args.add_stay_number:
        logger.info(f'Adding stay number to {name}')
        #df = pl.scan_parquet(data_path / out_name)
        #out_name = out_name.replace('.parquet', '_rn.parquet')
        ##df = df.group_by('ppn_int').map_groups(lambda group: group.with_columns(pl.arange(0, pl.count()).alias('stay_number')))
        #df = df.sort('episode_start_date').with_columns(stay_number=pl.col('ppn_int').cum_count().over('ppn_int'))
        #df.collect(streaming=True).write_parquet(data_path / out_name, use_pyarrow=True)
        name = out_name
        out_name = out_name.replace('.parquet', '_rn.parquet')
        query = f"""
        COPY (
        SELECT *, row_number() OVER(PARTITION BY ppn_int ORDER BY episode_start_date) as stay_number FROM
            read_parquet('{data_path}/{name}')
        ) to '{data_path}/{out_name}' (FORMAT 'PARQUET', CODEC 'ZSTD');
        """
        duckdb.sql(query)


if __name__ == '__main__':
    main()
