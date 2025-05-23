import duckdb
import polars as pl
import polars.selectors as cs
from pathlib import Path
from tqdm import tqdm
from melt_procedures import melt_apdc_procedures
import re


def preprocess(df):
    df = df.with_columns(pl.col('^.*date$').str.to_date('%m/%d/%Y'))
    df = df.drop(cs.matches('(procedure_location)|(condition_onset_flag)'))
    df = df.drop(cs.matches('(apdcflag)|(location)'))
    df = df.with_columns(pl.col('hospital_type').replace({4:2}))
    return df

def main():
    data_path = Path('/mnt/data_volume/apdc/study1/preprocessed')
    streaming = True
    groups = 50

    '''
    #df = pl.scan_parquet(f'{data_path}/_group_*.parquet')
    df = pl.scan_parquet(f'{data_path}/_group_*.parquet')
    column_names = df.columns
    pattern = r"(procedure.*|condition_onset_flag.*|apdcflag.*|diagnosis.*|block.*)"
    column_names = [c for c in column_names if not re.search(pattern, c)]
    df = df.select(pl.col(column_names).shrink_dtype())
    df.collect(streaming=True).write_parquet('full_cohort4.parquet')
    return
    column_names = ','.join(column_names)
    duckdb.execute("PRAGMA enable_progress_bar;")
    #df = df.drop(cs.matches('(procedure_location)|(condition_onset_flag)'))
    #df = df.drop(cs.matches('(apdcflag)|(location)'))
    #duckdb.execute("PRAGMA threads=4;")
    #duckdb.execute("PRAGMA temp_directory='/mnt/fast_drive/temp';")
    name= 'full_cohort3'
    files = [f'{data_path}/_group_{i}.parquet' for i in range(groups)]
    duckdb.sql(f"COPY (select {column_names} from read_parquet({files})) to '{data_path}/{name}.parquet' (FORMAT 'PARQUET', CODEC 'ZSTD');")
    #duckdb.sql(rf"COPY (select * from read_parquet({files})) to '{data_path}/{name}.parquet' (FORMAT 'PARQUET', CODEC 'ZSTD');")
    return
    '''
    duckdb.sql(f"COPY (select * from read_parquet({files})) to '{data_path}/{name}.parquet' (FORMAT 'PARQUET', CODEC 'ZSTD');")
    return
    df = pl.scan_parquet(f'{data_path}/full_cohort.parquet')
    df = df.with_columns(
            pl.col('episode_end_date').cast(pl.Datetime) 
                    + pl.duration(seconds=pl.col('episode_end_time'))
        )
    df.sink_parquet(f'{data_path}/full_cohort_2.parquet')
    return

    for i in range(groups):
        cohort = pl.scan_parquet(data_path / f'group_{i}.parquet')
        cohort = cohort.drop(cs.matches('(procedure_location)|(condition_onset_flag)'))
        cohort = cohort.drop(cs.matches('(apdcflag)|(location)'))
        #cohort = cohort.with_columns(pl.all().shrink_dtype())

        # add time to date to make date-time column
        cohort = cohort.with_columns(
            pl.col('episode_start_date').cast(pl.Datetime) 
                    + pl.duration(seconds=pl.col('episode_start_time')),
            pl.col('episode_end_date').cast(pl.Datetime) 
                    + pl.duration(seconds=pl.col('episode_end_time'))
        )
        cohort = cohort.with_columns(
            # add 1 second cumulatively to every row of a patient to make sure no two episodes
            # fall on the same exact date and time
            (pl.col('episode_start_date')+pl.duration(seconds=pl.int_range(pl.len()))).over('ppn_int')
        )

        # melt procedures and save
        df_procedure_block, df_procedure_code = melt_apdc_procedures(cohort)

        # save
        df_procedure_block.collect(streaming=streaming).write_parquet(data_path / f'group_{i}_procedure_block.parquet')
        df_procedure_code.collect(streaming=streaming).write_parquet(data_path / f'group_{i}_procedure_code.parquet')

        # save static data
        cohort_static = cohort.group_by('ppn_int').first()
        out_name_static = data_path / f'group_{i}_static.parquet'
        cohort_static.collect(streaming=streaming).write_parquet(out_name_static)

        cohort.collect(streaming=streaming).write_parquet(f'{data_path}/_group_{i}.parquet')

    # merge various group files into single files
    # procedure blocks
    print('Saving procedure blocks')
    df_proc = pl.scan_parquet(f'{data_path}/group_*_procedure_block.parquet')
    df_proc.sink_parquet(f'{data_path}/full_cohort_procedure_block.parquet')

    print('Saving procedure codes')
    # procedure codes
    df_proc = pl.scan_parquet(f'{data_path}/group_*_procedure_code.parquet')
    df_proc.sink_parquet(f'{data_path}/full_cohort_procedure_code.parquet')

    print('Saving static data')
    # static data
    df_static = pl.scan_parquet(f'{data_path}/group_*_static.parquet')
    df_static.with_columns(pl.all().shrink_dtype()).collect(streaming=True).write_parquet(f'{data_path}/full_cohort_static3.parquet')

    duckdb.execute("PRAGMA enable_progress_bar;")
    duckdb.execute("PRAGMA temp_directory='/mnt/fast_drive/temp';")
    name= 'full_cohort'
    files = [f'{data_path}/_group_{i}.parquet' for i in range(groups)]
    duckdb.sql(f"COPY (select * from read_parquet({files})) to '{data_path}/{name}.parquet' (FORMAT 'PARQUET', CODEC 'ZSTD');")

    return
    print('Saving all cohort data')
    cohort = pl.scan_parquet(f'{data_path}/_group_*.parquet')
    cohort.sink_parquet(f'{data_path}/full_cohort2.parquet')
    return


    duckdb.execute("PRAGMA enable_progress_bar;")
    df = df.drop(cs.matches('(procedure_location)|(condition_onset_flag)'))
    df = df.drop(cs.matches('(apdcflag)|(location)'))
    #duckdb.execute("PRAGMA threads=4;")
    #duckdb.execute("PRAGMA temp_directory='/mnt/fast_drive/temp';")
    name= 'full_cohort'
    files = [f'{data_path}/group_{i}.parquet' for i in range(groups)]
    duckdb.sql(f"COPY (select * from read_parquet({files})) to '{data_path}/{name}.parquet' (FORMAT 'PARQUET', CODEC 'ZSTD');")
    return

    print('Saving all cohort data')
    # all data
    cohort = pl.scan_parquet([f'{data_path}/group_{i}.parquet' for i in range(groups)])
    cohort.sink_parquet(f'{data_path}/full_cohort.parquet')

    return
    duckdb.execute("PRAGMA enable_progress_bar;")
    #duckdb.execute("PRAGMA threads=4;")
    #duckdb.execute("PRAGMA temp_directory='/mnt/fast_drive/temp';")
    name= 'full_cohort_procedure_block'
    duckdb.sql(f"COPY (select * from read_parquet('{data_path}/group_*_procedure_block.parquet')) to '{data_path}/{name}.parquet' (FORMAT 'PARQUET', CODEC 'ZSTD');")

    name= 'full_cohort_procedure_code'
    duckdb.sql(f"COPY (select * from read_parquet('{data_path}/group_*_procedure_code.parquet')) to '{data_path}/{name}.parquet' (FORMAT 'PARQUET', CODEC 'ZSTD');")


if __name__ == '__main__':
    main()
