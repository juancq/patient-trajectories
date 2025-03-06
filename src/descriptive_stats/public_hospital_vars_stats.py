# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:50:27 2023

@author: juanqa
"""
import os
import polars as pl 
from pathlib import Path


def main():
    root_path = Path('/mnt/data_volume/apdc')

    #apdc_eddc_path = root_path / 'str_group_files'
    apdc_eddc_path = root_path / 'study1'
    pl.Config.set_tbl_rows(100)
    apdc = pl.scan_parquet(apdc_eddc_path / 'cohort_apdc_group_*_data.parquet')
    apdc = apdc.with_columns(pl.col(['finsep', 'episode_start_date', 'episode_end_date']).str.to_date(format='%m/%d/%Y'))

    print(apdc.select(pl.col(['age_recode', 'EPISODE_OF_CARE_TYPE']).is_null().sum()).collect())
    print(apdc.select(pl.col(['EPISODE_LENGTH_OF_STAY']).is_null().sum()).collect())

    apdc = apdc.filter(pl.col('hospital_type')>1)
    apdc = apdc.with_columns(
        pl.when((pl.col('hospital_type') > 1) & (pl.col('PEER_GROUP').is_null()))
        .then(pl.lit('PRIV'))
        .otherwise(pl.col('PEER_GROUP'))
        .alias('PEER_GROUP')
        )
    for col in ['PEER_GROUP', 'ACUTE_FLAG', 'EMERGENCY_STATUS_RECODE', 'RECOGNISED_PH_FLAG']:
        print(apdc.select(pl.col(col).value_counts(sort=True)).unnest(col).collect(streaming=True))
        
        
if __name__ == '__main__':
    main()
