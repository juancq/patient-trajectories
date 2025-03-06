# -*- coding: utf-8 -*-
"""
@author: juanqa
"""
import os
import polars as pl 
from pathlib import Path
from datetime import datetime


def episodes_after_date(df, col, date):
    df = df.filter(pl.col(col) > date)
    after_threshold_count = df.select(pl.len()).collect().item()
    print(f'Episodes with {col} after {date}: {after_threshold_count}')
    df = df.with_columns(pl.col(col).dt.month().alias('month'))
    groups = df.group_by('month').agg(
        pl.len(),
        pl.col('totlos').mean(),
    )
    print(groups.sort('month').collect())


def main():
    '''
    Calculates dates stats:

    Sample output on all data:
    APDC
    Max admission date public hospital  2022-09-30
    Max discharge date public hospital  2022-11-17
    Max admission date private hospital 2021-07-06
    Max discharge date private hospital 2021-06-30
    EDDC
    Max admission date ED 2022-09-30
    Max discharge date ED 2022-11-17
    '''

    root_path = Path('/mnt/data_volume/apdc')
    apdc_eddc_path = root_path / 'str_group_files'

    # ------------------------------------------------------
    # apdc stats
    apdc = pl.scan_parquet(apdc_eddc_path / 'apdc_group_*.parquet')
    apdc = apdc.with_columns(pl.col(['finsep', 'episode_start_date', 'episode_end_date']).str.to_date('%m/%d/%Y'))

    print('-'*30)
    print('APDC')
    public = apdc.filter(pl.col('hospital_type')==1)
    private = apdc.filter(~(pl.col('hospital_type')==1))
    print('Min-Max admission ', apdc.select(
        pl.col('episode_start_date').min().alias('min'),pl.col('episode_start_date').max().alias('max')).collect())
    print('Max admission date public hospital ', public.select(pl.col('episode_start_date').max()).collect().item())
    print('Max discharge date public hospital ', public.select(pl.col('episode_end_date').max()).collect().item())
    print('Max admission date private hospital', private.select(pl.col('episode_start_date').max()).collect().item())
    print('Max discharge date private hospital', private.select(pl.col('episode_end_date').max()).collect().item())

    date_threshold = datetime(2022, month=6, day=30)
    episodes_after_date(apdc, 'finsep', date_threshold)
    episodes_after_date(apdc, 'episode_start_date', date_threshold)

    date_threshold = datetime(2022, month=9, day=30)
    episodes_after_date(apdc, 'finsep', date_threshold)
    episodes_after_date(apdc, 'episode_start_date', date_threshold)


    # ------------------------------------------------------
    # eddc stats
    print('-'*30)
    print('EDDC')
    ed = pl.scan_parquet(apdc_eddc_path / 'eddc_group_*.parquet')
    # rename eddc date/time columns to same name as apdc
    ed = ed.with_columns(pl.col('arrival_date').alias('episode_start_date'))
    ed = ed.with_columns(pl.col('actual_departure_date').alias('episode_end_date'))
    ed = ed.with_columns(pl.col(['episode_start_date', 'episode_end_date']).str.to_date('%m/%d/%Y'))
    print('Min admission ', ed.select(pl.col('episode_start_date').min()).collect().item())
    print('Max admission date ED', public.select(pl.col('episode_start_date').max()).collect().item())
    print('Max discharge date ED', public.select(pl.col('episode_end_date').max()).collect().item())

    ed = ed.with_columns(((pl.col('episode_end_date') - pl.col('episode_start_date')).dt.total_days()+1).alias('totlos'))
    date_threshold = datetime(2022, month=6, day=30)
    episodes_after_date(ed, 'episode_start_date', date_threshold)

    date_threshold = datetime(2022, month=9, day=30)
    episodes_after_date(ed, 'episode_start_date', date_threshold)

    # ------------------------------------------------------
    # death stats
    print('-'*30)
    print('RBDM')
    rbdm = pl.scan_parquet('/mnt/data_volume/apdc/rbdm_processed.parquet')
    print('Min date', rbdm.select(pl.col('DEATH_DATE').min()).collect().item())
    print('Max date', rbdm.select(pl.col('DEATH_DATE').max()).collect().item())

    print('-'*30)
    print('COD URF')
    cod = pl.scan_parquet('/mnt/data_volume/apdc/codurf_processed.parquet')
    print('Min date', cod.select(pl.col('DEATH_DATE').min()).collect().item())
    print('Max date', cod.select(pl.col('DEATH_DATE').max()).collect().item())

    # ------------------------------------------------------
    # cost stats
    print('-'*30)
    print('APDC ABF DNR')
    abf_dnr = pl.scan_parquet(root_path / 'apdc*dnr*.parquet')
    recnum = 'apdc_project_recnum'
    #print(abf_dnr.columns)
    abf_dnr = abf_dnr.select([recnum, 'TOTAL']).with_columns(pl.col('TOTAL').cast(pl.Float32))
    abf_dnr = abf_dnr.join(apdc, on=recnum, coalesce=True)
    print('Min date', abf_dnr.select(pl.col('episode_start_date').min()).collect().item())
    print('Max date', abf_dnr.select(pl.col('episode_start_date').max()).collect().item())

    # ------------------------------------------------------
    # cost stats
    print('-'*30)
    print('EDDC ABF DNR')
    abf_dnr = pl.scan_parquet(root_path / 'eddc*dnr*.parquet')
    recnum = 'eddc_project_recnum'
    #print(abf_dnr.columns)
    abf_dnr = abf_dnr.select([recnum, 'TOTAL']).with_columns(pl.col('TOTAL').cast(pl.Float32))
    abf_dnr = abf_dnr.join(ed, on=recnum, coalesce=True)
    print('Min date', abf_dnr.select(pl.col('episode_start_date').min()).collect().item())
    print('Max date', abf_dnr.select(pl.col('episode_start_date').max()).collect().item())

        
if __name__ == '__main__':
    main()
