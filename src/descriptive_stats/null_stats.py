# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:50:27 2023

@author: juanqa
"""
import os
import polars as pl 
from pathlib import Path


def main():
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)

    root_path = Path('/mnt/data_volume/apdc')
    apdc_eddc_path = root_path / 'str_group_files'

    # ------------------------------------------------------
    # apdc stats
    apdc = pl.scan_parquet(apdc_eddc_path / 'apdc_group_*_str.parquet')
    apdc = apdc.rename(lambda column_name: column_name.lower())

    print('APDC')
    streaming_flag = True
    num_episodes = apdc.select(pl.len()).collect(new_streaming=streaming_flag).item()
    print('Number of APDC episodes: ', num_episodes)

    cols = ['ppn_int', 'age_recode', 'episode_start_time', 'episode_start_date', 
            'episode_end_date', 'episode_end_time']

    # number of null values
    print('Null values')
    print(apdc.select(pl.col(cols).is_null().sum()).collect(streaming=streaming_flag))
    # time is in seconds, max should be < 86400 seconds (24 hours)
    print(apdc.select(pl.col(['episode_start_time', 'episode_end_time']).cast(pl.Int32).max()).collect())
    print(apdc.select(pl.col(['episode_start_time', 'episode_end_time']).cast(pl.Int32).min()).collect())
    print(apdc.select(pl.col(['episode_start_time', 'episode_end_time'])).collect().describe())
    print(apdc.select(pl.col(['episode_start_time', 'episode_end_time']).quantile(0.95)).collect())
    print(apdc.select(pl.col(['episode_start_time', 'episode_end_time']).quantile(0.999)).collect())

    common_cols = ['sex']
    apdc_cols = []
    apdc_cols = ['hospital_type', 'emergency_status_recode', 'episode_of_care_type',
                    'mode_of_separation_recode', 'peer_group', 
                    'facility_type', 'acute_flag']
    apdc_cols = apdc_cols + [
                    'facility_identifier_e',
                    'facility_trans_from_e',
                    'facility_trans_to_e',
                    'referred_to_on_separation_recode',
                    'ed_status',
                    'source_of_referral_recode',
    ]

    categorical = ['sex', 'hospital_type', 'emergency_status_recode', 'episode_of_care_type',
                    'mode_of_separation_recode', 'peer_group', 'mthyr_birth'] 
    print(apdc.select(pl.len(), pl.col(categorical).is_null().sum()).collect())
    print(apdc.select(pl.len(), pl.col(apdc_cols).is_null().sum()).collect())


    # ------------------------------------------------------
    # eddc stats
    print('EDDC')
    # read eddc 
    ed_cols = ['ppn_int', 'age_recode',  'sex',
                'episode_start_time', 'episode_start_date',
                'episode_end_time', 'episode_end_date',
                'departure_ready_time', 'departure_ready_date', 
                'triage_date', 'triage_time']
    ed = pl.scan_parquet(apdc_eddc_path / 'eddc_group_*_str.parquet')
    ed = ed.rename(lambda column_name: column_name.lower())
    # rename eddc date/time columns to same name as apdc
    ed = ed.with_columns(pl.col('arrival_date').alias('episode_start_date'))
    ed = ed.with_columns(pl.col('arrival_time').alias('episode_start_time'))
    ed = ed.with_columns(pl.col('actual_departure_date').alias('episode_end_date'))
    ed = ed.with_columns(pl.col('actual_departure_time').alias('episode_end_time'))
    # number of epdc records
    num_eddc_episodes = ed.select(pl.len()).collect().item()
    print('Number of EDDC episodes: ', num_eddc_episodes)
    print('Null count in EDDC:')
    print(ed.select(pl.len(), pl.col(ed_cols).is_null().sum()).collect())

    print(ed.columns)
    ed_cols = [ 'facility_identifier',
                'referred_to_on_departure_recode',
                'ed_source_of_referral',
                'ed_visit_type',
                'facility_type',
                'mode_of_arrival',
                'mode_of_separation',
                'peer_group', 
                'triage_category', 
                'sex', 
                'mthyr_birth_date', 
                ]
    print(ed.select(pl.len(), pl.col(ed_cols).is_null().sum()).collect())

    # ------------------------------------------------------
    # abf drn stats
    print('ABF DNR')
    # read cost data
    abf_dnr = pl.scan_parquet(root_path / 'apdc*dnr*.parquet')
    abf_dnr = abf_dnr.select(['apdc_project_recnum', 'TOTAL']).with_columns(pl.col('TOTAL').cast(pl.Float32))
    print('Null values')
    print(abf_dnr.select(pl.col('TOTAL').is_null().sum()).collect())
    print('Nan values')
    print(abf_dnr.select(pl.col('TOTAL').is_nan().sum()).collect())

        
if __name__ == '__main__':
    main()
