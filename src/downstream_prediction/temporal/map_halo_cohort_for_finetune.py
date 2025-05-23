import argparse
from pathlib import Path
import pickle
import numpy as np
import polars as pl
from tqdm import tqdm
from loguru import logger

## map_apdc_data file in parent directory, copying it to local folder for now
from map_apdc_data import map_data


pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(20)

streaming_flag = False

def main():
    logger.info('Mapping data for HALO')
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_file', type=str, help='index file', default=None)
    parser.add_argument('--cohort', type=str, help='input file', default=None)
    parser.add_argument('--output', type=str, help='output file', default=None)
    args = parser.parse_args()

    df = pl.scan_parquet(Path(args.cohort))

    n_unique = df.select(pl.col('ppn_int').n_unique()).collect().item()
    len_ = df.select(pl.len()).collect().item()
    logger.info(f'{args.cohort} unique ids and len: {n_unique}, {len_}')

    index_date = pl.scan_parquet(Path(args.index_file))
    n_unique = index_date.select(pl.col('ppn_int').n_unique()).collect().item()
    len_ = index_date.select(pl.len()).collect().item()
    logger.info(f'Label unique ids and len: {n_unique}, {len_}')
    df_window = df.join(index_date, on='ppn_int', coalesce=True)
    
    df_window = df_window.filter(
        # make sure episode_start_date is a date, without time
        (pl.col('episode_start_date').cast(pl.Date) <= pl.col('index_date'))
        .over('ppn_int')
    )

    n_unique = df_window.select(pl.col('ppn_int').n_unique()).collect().item()
    len_ = df_window.select(pl.len()).collect().item()
    logger.info(f'df window after filter unique ids and len: {n_unique}, {len_}')

    df_vars = pl.read_excel("temporal_df_vars.xlsx")
    variable_mapping_fname = 'temporal_variable_mapping_for_halo.pkl'

    if df_vars is None:
        raise ValueError("df_vars.xlsx is missing")
    if len(df_vars) == 0:
        raise ValueError("df_vars.xlsx is empty")

    variable_mapping_fname = Path(variable_mapping_fname)
    if not variable_mapping_fname.exists() or variable_mapping_fname.is_dir():
        raise FileNotFoundError(f"Path does not exist: {variable_mapping_fname}")

    variable_mapping = pickle.load(open(variable_mapping_fname, "rb"))

    df_mapped = map_data(df_window, df_vars, variable_mapping)

    print('Processing ...')
    df_mapped = df_mapped.collect(streaming=streaming_flag)
    # statistics
    num_visits = df_mapped.select(pl.col('visits').list.len())
    num_patients = len(df_mapped)
    print(f"Number of patients: {num_patients}")
    print(f"Maximum number of visits: {max(num_visits)}")
    print(f"Average number of visits: {np.mean(np.array(num_visits))}")

    print('Writing parquet')
    df_mapped.write_parquet(Path(args.output))


if __name__ == '__main__':
    main()