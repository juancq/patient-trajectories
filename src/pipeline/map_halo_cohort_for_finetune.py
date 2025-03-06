from pathlib import Path
import pickle
import numpy as np
import polars as pl
from tqdm import tqdm

from map_apdc_data import map_data


pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(20)

streaming_flag = False

def main():
    apdc_eddc_path = Path('/mnt/data_volume/apdc/study1')

    project_path = apdc_eddc_path / 'preprocessed' 
    split = 'tuning'
    df = pl.scan_parquet(project_path / f'full_cohort_{split}.parquet')

    # prediction point path for various tasks
    index_date_path = project_path / 'tasks'
    # can use any of the task files, because the index date is the same across all
    # (this means index date should be stored separately)
    windows = ['six_months']#, 'one_year', 'two_year']
    for window in tqdm(windows):
        index_date = pl.scan_parquet(index_date_path / f'length_of_stay_{window}.parquet')
        df_window = df.join(index_date, on='ppn_int', coalesce=True)
        df_window = df_window.filter(
            (pl.col('episode_start_date') <= pl.col('index_date'))
            .over('ppn_int')
        )

        ## filter to one episode
        #df_window = df_window.filter((pl.len()==1).over('ppn_int'))
        ##df_window = df_window.filter((pl.col('episode_start_date')==pl.col('episode_end_date')).over('ppn_int'))
        #df_window = df_window.select(pl.col(['ppn_int', 'episode_start_date', 'episode_end_date', 'apdc', 'hospital_type']))
        #df_window.collect(streaming=True).write_csv('hello.csv')
        #return

        df_mapped = map_data(df_window)

        print('Processing ...')
        df_mapped = df_mapped.collect(streaming=streaming_flag)
        # statistics
        num_visits = df_mapped.select(pl.col('visits').list.len())
        num_patients = len(df_mapped)
        print(f"Number of patients: {num_patients}")
        print(f"Maximum number of visits: {max(num_visits)}")
        print(f"Average number of visits: {np.mean(np.array(num_visits))}")

        print('Writing parquet')
        #df_mapped.write_parquet(index_date_path / f"df_{split}_halo_{window}_v2.parquet")
        #df_mapped.write_parquet(index_date_path / f"df_{split}_halo_{window}_single_episode.parquet")
        df_mapped.write_parquet(index_date_path / f"df_{split}_halo_{window}_same_day.parquet")


if __name__ == '__main__':
    main()