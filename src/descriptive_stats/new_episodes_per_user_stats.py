# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:50:27 2023

@author: juanqa
"""
import glob
import polars as pl
from tqdm import tqdm


def stratify_episode_counts(df, key):
    print('by ', key)
    total = 0
    for i in [1, 2, 3]:
        count = df.filter((pl.col(key) == i)).select(pl.len()).collect().item()
        print(f'={i}, {count}')
        total += count

    df = df.filter(pl.col(key) > 3)
    count = df.filter(pl.col(key).is_between(4, 10, closed='left')).select(pl.len()).collect().item()
    print(f'[4,10), {count}')
    total += count

    count = df.filter(pl.col(key).is_between(10, 20, closed='left')).select(pl.len()).collect().item()
    print(f'[10,20), {count}')
    total += count

    count = df.filter(pl.col(key).is_between(20, 50, closed='left')).select(pl.len()).collect().item()
    print(f'[20,50), {count}')
    total += count

    count = df.filter(pl.col(key) >= 50).select(pl.len()).collect().item()
    print(f'50+, {count}')

    count = df.filter(pl.col(key).is_between(50, 128, closed='left')).select(pl.len()).collect().item()
    print(f'[50,128), {count}')
    total += count

    count = df.filter(pl.col(key).is_between(128, 256, closed='left')).select(pl.len()).collect().item()
    print(f'[128,256), {count}')
    total += count

    count = df.filter(pl.col(key) >= 256).select(pl.len()).collect().item()
    print(f'256+, {count}')
    total += count


def sex_stats(df):
    #df = df.select(pl.col(['SEX', 'age_recode', 'ppn_int', 'episode_start_date']))
    # first presentation
    df_first = df.filter((pl.col('episode_start_date')==pl.col('episode_start_date').min()).over('ppn_int'))
    df_first = df_first.group_by('ppn_int').first()
    # last presentation
    df_last = df.filter((pl.col('episode_start_date')==pl.col('episode_start_date').max()).over('ppn_int'))
    df_last = df_last.group_by('ppn_int').first()

    # gender information
    print('gender information')
    values = df_first.select(pl.col('SEX').value_counts(sort=True)).unnest('SEX').collect()
    print(values)

    print('backward fill')
    df_impute = df.with_columns(pl.col('SEX').backward_fill().over('ppn_int'))
    df_impute = df_impute.filter((pl.col('episode_start_date')==pl.col('episode_start_date').min()).over('ppn_int'))
    df_impute = df_impute.group_by('ppn_int').first()
    values = df_impute.select(pl.col('SEX').value_counts(sort=True)).unnest('SEX').collect()
    print(values)

    df_impute = df_impute.filter(~pl.col('SEX').is_in(['1', '2']))
    df = df.join(df_impute.select('ppn_int'), on='ppn_int', how='anti')
    print('cohort size check:')
    print(df.select(pl.col('ppn_int').n_unique()).collect().item())
    return df


def age_stats(df):
    #df = df.select(pl.col(['SEX', 'age_recode', 'ppn_int', 'episode_start_date']))
    # first presentation
    df_first = df.filter((pl.col('episode_start_date')==pl.col('episode_start_date').min()).over('ppn_int'))
    df_first = df_first.group_by('ppn_int').first()
    # last presentation
    df_last = df.filter((pl.col('episode_start_date')==pl.col('episode_start_date').max()).over('ppn_int'))
    df_last = df_last.group_by('ppn_int').first()

    # gender information
    print('gender information')
    values = df_first.select(pl.col('SEX').value_counts(sort=True)).unnest('SEX').collect()
    print(values)

    print('backward fill')
    df_impute = df.with_columns(pl.col('SEX').backward_fill().over('ppn_int'))
    df_impute = df_impute.filter((pl.col('episode_start_date')==pl.col('episode_start_date').min()).over('ppn_int'))
    df_impute = df_impute.group_by('ppn_int').first()
    values = df_impute.select(pl.col('SEX').value_counts(sort=True)).unnest('SEX').collect()
    print(values)

    print('Under 6 years old at first episode')
    print(df_first.filter(pl.col('age_recode') <6).select(pl.len()).collect().item())

    print('Under 6 years old at last episode')
    under_6 = df_last.filter(pl.col('age_recode') <6)
    print(under_6.select(pl.len()).collect().item())

    print('age brackets at last presentation')
    total = 0
    count = df_last.filter(pl.col('age_recode').is_between(6, 10)).select(pl.len()).collect().item()
    total += count
    print(f'[6, 10] = {count}')
    df_last = df_last.filter(pl.col('age_recode') > 10)
    step = 10
    for i in range(10, 100, step):
        lower = i
        upper = lower + step
        if i < 90:
            count = df_last.filter(pl.col('age_recode').is_between(i, i+step, closed='right')).select(pl.len()).collect().item()
            print(f'({lower}, {upper}] = {count}')
        else:
            count = df_last.filter(pl.col('age_recode') > 90).select(pl.len()).collect().item()
            print(f'>90 = {count}')
        total += count
    print(f'cohort size after remove < 6 = {total}')

    df = df.join(under_6.select('ppn_int'), on='ppn_int', how='anti')
    print('cohort size check:')
    print(df.select(pl.col('ppn_int').n_unique()).collect().item())
    return df


def main():
    path = 'H:/apdc/str_group_files'
    
    pl.Config.set_tbl_rows(100)
    apdc = pl.scan_parquet(f'{path}/apdc_group_*.parquet')
    apdc_ppn = apdc.select(pl.col(['ppn_int']))

    ed = pl.scan_parquet(f'{path}/eddc_group_*.parquet')
    ed = ed.with_columns(pl.col('arrival_date').alias('episode_start_date'))
    ed_ppn = ed.select(pl.col(['ppn_int']))

    # remove zombies - episodes after death
    cod = pl.scan_parquet('H:/apdc/codurf_processed.parquet')

    # apdc and eddc combined
    df = pl.concat([apdc, ed], how='diagonal')
    df = df.with_columns(pl.col('episode_start_date').str.to_datetime('%m/%d/%Y'))

    last_episode = df.group_by('ppn_int').agg(pl.col('episode_start_date').max())
    last_episode = last_episode.join(cod, on='ppn_int')
    zombies = last_episode.filter(pl.col('DEATH_DATE') < pl.col('episode_start_date'))
    num_zombies = zombies.select(pl.len()).collect().item()
    print(f'zombies = {num_zombies}')

    # remove zombies
    df = df.join(zombies.select('ppn_int'), on='ppn_int', how='anti')

    # get age stats, removes < 6 years of age
    df = age_stats(df)

    # get sex stats, removes sex other than male/female?
    df = sex_stats(df)

    # number of episodes
    apdc_result = df.group_by('ppn_int').len()
    print(apdc_result.select('len').describe())
    stratify_episode_counts(apdc_result, 'len')

    single_episode = apdc_result.filter(pl.col('len') == 1)

    df = df.join(single_episode, on='ppn_int', how='anti')
    print(df.select(pl.col('ppn_int').n_unique()).collect().item())

    # get age stats, removes < 6 years of age
    age_stats(df)
        
        
if __name__ == '__main__':
    main()
