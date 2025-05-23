'''
Splits entire cohort into a pretrain cohort and a downstream cohort using a temporal split.
'''
import duckdb
import polars as pl
import polars.selectors as cs
from pathlib import Path
from tqdm import tqdm
from datetime import date


def preprocess(df):
    df = df.with_columns(pl.col('^.*date$').str.to_date('%m/%d/%Y'))
    df = df.drop(cs.matches('(procedure_location)|(condition_onset_flag)'))
    df = df.drop(cs.matches('(apdcflag)|(location)'))
    df = df.with_columns(pl.col('hospital_type').replace({4:2}))
    return df

def main():
    data_path = Path('/mnt/data_volume/apdc/study1/preprocessed')
    streaming = {'engine': 'streaming'}

    # --- Parameters ---
    PATIENT_ID_COL = "ppn_int"
    DATE_COL = "episode_end_date"
    RANDOM_SEED = 42 # For reproducible splits
    groups = 50

    # Date cutoffs
    PRETRAIN_END_DATE = date(2017, 12, 31)
    # to be used in a separate script that splits downstream cohort
    #DOWNSTREAM_TRAIN_START_DATE = date(2016, 1, 1)
    #DOWNSTREAM_TRAIN_END_DATE = PRETRAIN_END_DATE
    #DOWNSTREAM_TEST_START_DATE = date(2018, 1, 1)
    #DOWNSTREAM_TEST_END_DATE = date(2019, 12, 31)

    # Minimum episodes for pretrain set
    MIN_PRETRAIN_EPISODES = 2

    for i in tqdm(range(groups)):
        cohort = pl.scan_parquet(data_path / f'group_{i}.parquet')
        cohort = cohort.drop(cs.matches('(procedure_location)|(condition_onset_flag)'))
        cohort = cohort.drop(cs.matches('(apdcflag)|(location)'))
        #cohort = cohort.with_columns(pl.all().shrink_dtype())

        # operations to determine temporal split
        departure_cols = ['actual_departure_date', 'departure_ready_date']
        cohort = cohort.with_columns(
            pl.col('episode_start_date').cast(pl.Date).alias('episode_start_date_strict'),
            pl.col('episode_end_date').cast(pl.Date),
            pl.col(departure_cols).str.to_date("%m/%d/%Y"),
        )

        # set episode end date to at least the start date if end_date has an error
        # (very small number of records)
        cohort = cohort.with_columns(
            pl.when(pl.col('episode_end_date') < pl.col('episode_start_date_strict'))
            # this should be checked on the preprocessing
            .then(pl.max_horizontal('episode_start_date_strict', *departure_cols))
            .otherwise(pl.col('episode_end_date'))
            .alias('episode_end_date')
        )

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


        # keep only episodes that end before pretrain end date
        pretrain_cohort = cohort.filter(
            (pl.col('episode_end_date') <= PRETRAIN_END_DATE)
        )

        # count episodes per subject and keep those with at least 2 episodes
        pretrain_cohort = pretrain_cohort.filter(
            pl.len().over(PATIENT_ID_COL) >= MIN_PRETRAIN_EPISODES
        )

        pretrain_cohort.collect(**streaming).write_parquet(f'{data_path}/temporal_pretrain_group_{i}.parquet')

        downstream_cohort = cohort.join(pretrain_cohort, how='anti', on='ppn_int')
        downstream_cohort.collect(**streaming).write_parquet(f'{data_path}/temporal_downstream_group_{i}.parquet')

    # merge various group files into single files
    duckdb.execute("PRAGMA enable_progress_bar;")
    name= 'temporal_pretrain_full_cohort'
    files = [f'{data_path}/temporal_pretrain_group_{i}.parquet' for i in range(groups)]
    duckdb.sql(f"COPY (select * from read_parquet({files})) to '{data_path}/{name}.parquet' (FORMAT 'PARQUET', CODEC 'ZSTD');")

    name= 'temporal_downstream_full_cohort'
    files = [f'{data_path}/temporal_downstream_group_{i}.parquet' for i in range(groups)]
    duckdb.sql(f"COPY (select * from read_parquet({files})) to '{data_path}/{name}.parquet' (FORMAT 'PARQUET', CODEC 'ZSTD');")


if __name__ == '__main__':
    main()
