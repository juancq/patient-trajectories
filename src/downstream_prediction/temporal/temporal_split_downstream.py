'''
Splits downstream cohort into train/test.
'''

import argparse
import polars as pl
from pathlib import Path
from datetime import date
from loguru import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='input file')
    parser.add_argument('--output_path', type=str, help='input file')
    args = parser.parse_args()
    input_file = Path(args.input_file)
    #streaming = {'engine': 'streaming'}
    streaming_flag = False

    PATIENT_ID_COL = "ppn_int"
    DATE_COL = "episode_end_date"
    RANDOM_SEED = 42 # For reproducible splits
    PRETRAIN_END_DATE = date(2017, 12, 31)
    # to be used in a separate script that splits downstream cohort
    DOWNSTREAM_TRAIN_START_DATE = date(2016, 1, 1)
    DOWNSTREAM_TRAIN_END_DATE = date(2018, 12, 31)
    DOWNSTREAM_TEST_START_DATE = date(2018, 1, 1)
    DOWNSTREAM_TEST_END_DATE = date(2020, 1, 31)

    downstream_cohort = pl.scan_parquet(input_file)

    # Downstream Train DataFrame
    downstream_train = downstream_cohort.filter(
        # keeps only episodes through the end of training time
        (pl.col('episode_end_date') <= DOWNSTREAM_TRAIN_END_DATE)
    )

    # Downstream Test DataFrame
    downstream_test = downstream_cohort.join(
        downstream_train, on='ppn_int', how='anti'
    )

    downstream_test = downstream_test.filter(
        #(pl.col(DATE_COL) >= DOWNSTREAM_TEST_START_DATE) &
        (pl.col(DATE_COL) <= DOWNSTREAM_TEST_END_DATE)
    )

    logger.info(downstream_train.select(pl.len()).collect(streaming=streaming_flag))
    logger.info(downstream_test.select(pl.len()).collect(streaming=streaming_flag))

    logger.info(downstream_train.select(pl.col('ppn_int').n_unique()).collect(streaming=streaming_flag))
    logger.info(downstream_test.select(pl.col('ppn_int').n_unique()).collect(streaming=streaming_flag))

    downstream_train.select(pl.col('ppn_int').unique()).collect(streaming=streaming_flag).write_csv('downstream_train.csv')
    downstream_test.select(pl.col('ppn_int').unique()).collect(streaming=streaming_flag).write_csv('downstream_test.csv')

    result = downstream_train.group_by('ppn_int').len().collect(streaming=streaming_flag)
    logger.info(result.select('len').describe())

    result = downstream_test.group_by('ppn_int').len().collect(streaming=streaming_flag)
    logger.info(result.select('len').describe())

    min_date = downstream_train.select(pl.col('episode_end_date').min()).collect().item()
    max_date = downstream_train.select(pl.col('episode_end_date').max()).collect().item()
    logger.info(f"\nInitial dataset: {downstream_train.select(pl.len()).collect().item()} rows")
    logger.info(f"Date range: {min_date} to {max_date}")

    min_date = downstream_test.select(pl.col('episode_end_date').min()).collect().item()
    max_date = downstream_test.select(pl.col('episode_end_date').max()).collect().item()
    logger.info(f"\nInitial dataset: {downstream_test.select(pl.len()).collect().item()} rows")
    logger.info(f"Date range: {min_date} to {max_date}")

    output_path = Path(args.output_path)
    train_fname = output_path / 'temporal_downstream_train.parquet'
    test_fname = output_path / 'temporal_downstream_test.parquet'

    logger.info(f"Saving downstream train: {train_fname}") 
    downstream_train.sink_parquet(train_fname)

    logger.info(f"Saving downstream test: {test_fname}") 
    #downstream_test.sink_parquet(test_fname)
    downstream_test.collect(streaming=streaming_flag).write_parquet(test_fname)

    #downstream_train.collect(streaming=streaming_flag).write_parquet(data_path / 'temporal_downstream_train.parquet')
    return


if __name__ == '__main__':
    main()
