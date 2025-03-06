# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:50:27 2023

@author: juanqa
"""
import os
import polars as pl 
from pathlib import Path


ROOT_PATH = Path('/mnt/data_volume/apdc')


def main():
    root_path = Path('/mnt/data_volume/apdc')

    #apdc_eddc_path = root_path / 'str_group_files'
    apdc_eddc_path = root_path / 'study1'
    pl.Config.set_tbl_rows(100)

    eddc = pl.scan_parquet(apdc_eddc_path / 'cohort_eddc_group_*_data.parquet')
    eddc = eddc.with_columns(pl.col('arrival_date').alias('episode_start_date'))
    eddc = eddc.with_columns(pl.col('actual_departure_date').alias('episode_end_date'))
    eddc = eddc.with_columns(pl.col('episode_end_date').alias('finsep'))
    eddc = eddc.with_columns(pl.col(['finsep', 'episode_start_date', 'episode_end_date']).str.to_date(format='%m/%d/%Y'))

    # eddc
    print('EDDC')
    eddc = eddc.with_columns(((pl.col('episode_end_date') - pl.col('episode_start_date')).dt.total_days()+1).alias('totlos'))
    ed_num_episodes = eddc.select(pl.len()).collect().item()
    print('Number of EDDC episodes: ', ed_num_episodes)

    # eddc
    recnum = 'eddc_project_recnum'
    # read eddc cost data
    abf_dnr = pl.scan_parquet(root_path / 'eddc_abf_dnr*.parquet')
    abf_dnr.head().collect()
    abf_dnr = abf_dnr.select([recnum, 'TOTAL']).with_columns(pl.col('TOTAL').cast(pl.Float32))

    abf_dnr_misc = abf_dnr.join(eddc, on=recnum, coalesce=True)
    abf_dnr= abf_dnr_misc.with_columns(pl.col('episode_start_date').dt.year().alias('year'))

    # count null episodes by year

    ed = eddc
    # read eddc 
    ed_cols = ['ppn_int', 'age_recode', 'episode_start_time', 'episode_start_date', 'SEX']
    # rename eddc date/time columns to same name as apdc
    ed = ed.with_columns(pl.col('arrival_date').alias('episode_start_date'))
    ed = ed.with_columns(pl.col('arrival_time').alias('episode_start_time'))

    # number of epdc records
    num_eddc_episodes = ed.select(pl.len()).collect().item()
    print('Number of EDDC episodes: ', num_eddc_episodes)

    print('Cost Stats')
    # cost stats from abf dnr
    neg_cost = abf_dnr.filter(pl.col('TOTAL') < 0)
    num_neg_cost = neg_cost.select(pl.len()).collect().item()
    print(f'Episodes with negative cost: {num_neg_cost}')

    zero_cost = abf_dnr.filter(pl.col('TOTAL') < 1)
    num_zero_cost = zero_cost.select(pl.len()).collect().item()
    print(f'Episodes with zero cost: {num_zero_cost}')
    # less than $10
    low_cost_episodes = abf_dnr.filter(pl.col('TOTAL') < 10)
    num_low_cost = low_cost_episodes.select(pl.len()).collect().item()
    print(f'Episodes with cost < $10: {num_low_cost}')
    #low_cost_episodes.collect().write_csv('low_cost.csv')

    # less than $100
    low_cost_episodes = abf_dnr.filter(pl.col('TOTAL') < 100)
    num_low_cost = low_cost_episodes.select(pl.len()).collect().item()
    print(f'Episodes with cost < $100: {num_low_cost}')

    # join apdc to abf_dnr to calculate cost stats faster (dropping episodes without cost)
    abf_dnr_misc = abf_dnr.join(ed.select(pl.col(['eddc_project_recnum', 'episode_start_date', 'age_recode'])), on='eddc_project_recnum', coalesce=True)
    abf_dnr_misc = abf_dnr_misc.with_columns(pl.col('episode_start_date').dt.year().alias('year'))
    cost_episodes_num = abf_dnr_misc.select(pl.len()).collect().item()
    quantiles = [.25, .50, .75]
    null_groups = abf_dnr_misc.group_by('year').agg(
        pl.len().alias('N'),
        (pl.len()/cost_episodes_num).alias('%'),
        pl.col('TOTAL').mean(),
        *[pl.col('TOTAL').quantile(q).name.suffix(f'_q{q}') for q in quantiles],
        ((pl.col('age_recode') < 1).sum()/pl.len()).alias('under 1')
    )
    print(null_groups.sort('year').collect())

        
        
if __name__ == '__main__':
    main()
