# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:50:27 2023

@author: juanqa
"""
import os
import polars as pl 
from pathlib import Path


ROOT_PATH = Path('/mnt/data_volume/apdc')


def eddc_stats(df):
    ed_cols = ['ppn_int', 'age_recode',  'SEX'
                'episode_start_time', 'episode_start_date',
                'episode_end_time', 'episode_end_date',]
    # rename eddc date/time columns to same name as apdc
    df = df.with_columns(pl.col('arrival_date').alias('episode_start_date'))
    df = df.with_columns(pl.col('arrival_time').alias('episode_start_time'))
    df = df.with_columns(pl.col('actual_departure_date').alias('episode_end_date'))
    df = df.with_columns(pl.col('actual_departure_time').alias('episode_end_time'))
    df = df.with_columns(pl.col('episode_end_date').alias('finsep'))

    # number of epdc records
    num_eddc_episodes = df.select(pl.len()).collect().item()
    print('Number of EDDC episodes: ', num_eddc_episodes)
    print('Null count in EDDC:')
    print(df.select(pl.col(ed_cols).is_null().sum()).collect())

    abf_dnr = pl.scan_parquet(root_path / 'eddc*dnr*.parquet')
    abf_dnr = abf_dnr.select(['eddc_project_recnum', 'TOTAL']).with_columns(pl.col('TOTAL').cast(pl.Float32))

    abf_dnr_misc = abf_dnr.join(df, on='eddc_project_recnum', coalesce=True)
    abf_dnr= abf_dnr_misc.with_columns(pl.col('episode_start_date').str.to_datetime('%m/%d/%Y').dt.year().alias('year'))

    abf_dnr_2022 = abf_dnr.filter(pl.col('finsep') > (pl.date(2022, month=1, day=1)))
    print(abf_dnr_2022.select(pl.len()).collect().item())
    abf_dnr_2022 = abf_dnr_2022.with_columns(pl.col('finsep').dt.month().alias('month'))
    groups = abf_dnr_2022.group_by('month').len().collect()
    print(groups.sort('month'))



def main():
    root_path = Path('/mnt/data_volume/apdc')

    #apdc_eddc_path = root_path / 'str_group_files'
    apdc_eddc_path = root_path / 'study1'
    pl.Config.set_tbl_rows(100)
    apdc = pl.scan_parquet(apdc_eddc_path / 'cohort_apdc_group_*_data.parquet')
    apdc = apdc.with_columns(pl.col(['finsep', 'episode_start_date', 'episode_end_date']).str.to_date(format='%m/%d/%Y'))
    apdc = apdc.with_columns(pl.col('episode_start_date').dt.year().alias('year'))

    eddc = pl.scan_parquet(apdc_eddc_path / 'cohort_eddc_group_*_data.parquet')
    eddc = eddc.with_columns(pl.col('arrival_date').alias('episode_start_date'))
    eddc = eddc.with_columns(pl.col('actual_departure_date').alias('episode_end_date'))
    eddc = eddc.with_columns(pl.col('episode_end_date').alias('finsep'))
    eddc = eddc.with_columns(pl.col(['finsep', 'episode_start_date', 'episode_end_date']).str.to_date(format='%m/%d/%Y'))
    eddc = eddc.with_columns(pl.col('arrival_time').alias('episode_start_time'))

    num_episodes = apdc.select(pl.len()).collect().item()
    print('Number of APDC episodes: ', num_episodes)

    # eddc
    print('EDDC')
    eddc = eddc.with_columns(((pl.col('episode_end_date') - pl.col('episode_start_date')).dt.total_days()+1).alias('totlos'))
    ed_num_episodes = eddc.select(pl.len()).collect().item()
    print('Number of EDDC episodes: ', ed_num_episodes)

    print('-'*20)
    print('apdc abf dnr')
    # read apdc cost data
    abf_dnr = pl.scan_parquet(root_path / 'apdc*dnr*.parquet')
    recnum = 'apdc_project_recnum'
    abf_dnr = abf_dnr.select([recnum, 'TOTAL']).with_columns(pl.col('TOTAL').cast(pl.Float32))

    abf_dnr_misc = abf_dnr.join(apdc, on=recnum, coalesce=True)
    abf_dnr= abf_dnr_misc.with_columns(pl.col('episode_start_date').dt.year().alias('year'))

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

    # join apdc to abf_dnr to calculate cost stats faster (dropping episodes without cost)
    abf_dnr_misc = abf_dnr.join(apdc.select(pl.col(['apdc_project_recnum', 'episode_start_date', 'age_recode'])), on='apdc_project_recnum', coalesce=True)
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

    # join apdc data with ABF DNR
    apdc = apdc.join(abf_dnr, on='apdc_project_recnum', how='left', coalesce=True)

    print('Public and Private Episodes')
    apdc_null_cost = apdc.select(pl.col('TOTAL').is_null().sum()).collect().item()
    print(f'Episodes with null cost: {apdc_null_cost}')
    print(f'Episodes with cost: {num_episodes - apdc_null_cost}')
    print(abf_dnr.select(pl.col('TOTAL').is_null().sum()).collect().item())

    public_episodes = apdc.filter(pl.col('hospital_type')==1)
    num_public_episodes = public_episodes.select(pl.len()).collect().item()
    print(f'Public hospital episodes: {num_public_episodes}')

    # for public episodes
    print('Public Episodes')
    public_null_cost = public_episodes.select(pl.col('TOTAL').is_null().sum()).collect().item()
    print(f'Episodes with null cost: {public_null_cost}')

    public_episodes = public_episodes.with_columns(pl.col('episode_start_date').dt.year().alias('year'))
    null_groups = public_episodes.group_by('year').agg(
        pl.col('TOTAL').is_null().sum().name.suffix('_sum'),
        pl.len().alias('N'),
        (pl.col('TOTAL').is_null().sum()/pl.len()).alias('%')
    )
    print(null_groups.sort('year').collect())

    null_episodes = public_episodes.filter(pl.col('TOTAL').is_null())
    null_groups = null_episodes.group_by('year').agg(
        pl.len().alias('N'),
        *[pl.col('age_recode').quantile(q).name.suffix(f'_q{q}') for q in quantiles],
        (pl.col('SEX')=='2').sum()/pl.len(),
    )
    print(null_groups.sort('year').collect())


    # eddc
    print('-'*20)
    print('eddc abf dnr')
    recnum = 'eddc_project_recnum'
    # read eddc cost data
    abf_dnr = pl.scan_parquet(root_path / 'eddc*dnr*.parquet')
    abf_dnr.head().collect()
    abf_dnr = abf_dnr.select([recnum, 'TOTAL']).with_columns(pl.col('TOTAL').cast(pl.Float32))

    abf_dnr_misc = abf_dnr.join(eddc, on=recnum, coalesce=True)
    abf_dnr= abf_dnr_misc.with_columns(pl.col('episode_start_date').dt.year().alias('year'))

    # count null episodes by year

    # read eddc 
    ed_cols = ['ppn_int', 'age_recode', 'episode_start_time', 'episode_start_date', 'SEX']

    # number of epdc records
    num_eddc_episodes = eddc.select(pl.len()).collect().item()
    print('Number of EDDC episodes: ', num_eddc_episodes)
    print('Null count in EDDC:')
    print(eddc.select(pl.col(ed_cols).is_null().sum()).collect())

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

    # join apdc to abf_dnr to calculate cost stats faster (dropping episodes without cost)
    abf_dnr_misc = abf_dnr.join(eddc.select(pl.col(['eddc_project_recnum', 'episode_start_date', 'age_recode'])), on='eddc_project_recnum', coalesce=True)
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
