# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:50:27 2023

@author: juanqa
"""
import polars as pl


def main():
    '''
    Flow:
    1. remove those with episodes =1
    2. remove unqualified newborns
    3. remove zombies
    '''
    path = '/mnt/data_volume/apdc/str_group_files'
    
    # death date, for identifying zombies
    # have to use rbdm, since cod_urf is only available through the end of 2020
    death = pl.scan_parquet('/mnt/data_volume/apdc/rbdm_processed.parquet')

    pl.Config.set_tbl_rows(100)
    num_groups = 50
    total_count = 0
    for i in range(num_groups):

        apdc = pl.scan_parquet(f'{path}/apdc_group_{i}_str.parquet')
        apdc = apdc.with_columns(pl.lit(1).alias('apdc'))
        ed = pl.scan_parquet(f'{path}/eddc_group_{i}_str.parquet')
        ed = ed.with_columns(pl.col('arrival_date').alias('episode_start_date'))
        ed = ed.with_columns(pl.col('actual_departure_date').alias('episode_end_date'))
        ed = ed.with_columns(pl.lit(1).alias('eddc'))

        # apdc and eddc combined
        df = pl.concat([apdc, ed], how='diagonal')
        df = df.with_columns(pl.col(['episode_start_date', 'episode_end_date']).str.to_date('%m/%d/%Y'))

        # remove episodes after 2021-06-30 and before 2005-07-01
        # private hospital data not available after first week of july 2021
        end_of_followup = pl.date(year=2021, month=7, day=1)
        # ed data only available after this date
        start_of_followup = pl.date(year=2005, month=7, day=1)
        df = df.filter((pl.col('episode_end_date') < end_of_followup) & (pl.col('episode_start_date') >= start_of_followup))

        # remove ed only episodes from apdc
        ed_only_date = pl.date(year=2017, month=6, day=15)
        df = df.filter(
            ~((pl.col('episode_start_date') < ed_only_date) & (pl.col('ED_STATUS').is_in(['1','4'])))
        )

        # keep those with more than 1 episode
        df = df.filter((pl.len()>1).over('ppn_int'))

        df = df.with_columns(pl.col('SRG').fill_null(''))

        neonate_age_limit = 0.000001
        unqualified_neonate_expr = (pl.col('SRG') == '74') & (pl.col('age_recode') < neonate_age_limit)
        # drop birth episodes, which will also drop people whose only episode is birth
        df = df.filter(~unqualified_neonate_expr)

        # keep those with more than one episode
        df = df.filter((pl.len()>1).over('ppn_int'))

        # remove zombies - episodes after death
        last_episode = df.group_by('ppn_int').agg(pl.col('episode_start_date').max())
        last_episode = last_episode.join(death, on='ppn_int')
        zombies = last_episode.filter(pl.col('DEATH_DATE') < pl.col('episode_start_date'))

        # remove zombies
        df = df.join(zombies.select('ppn_int'), on='ppn_int', how='anti')

        # remove those without age information
        no_age = df.filter((pl.col('age_recode').is_null().all()).over('ppn_int'))
        df = df.join(no_age.select('ppn_int'), on='ppn_int', how='anti')

        # save ppns of cohort
        (df.select(pl.col('ppn_int').unique())
            .collect(streaming=True)
            .write_parquet(f'/mnt/data_volume/apdc/study1/cohort_ppn_group_{i}.parquet')
        )
        # save apdc ppn and episode id
        (df.filter(pl.col('apdc')==1).select(pl.col(['ppn_int','apdc_project_recnum']))
            .collect(streaming=True)
            .write_parquet(f'/mnt/data_volume/apdc/study1/cohort_apdc_group_{i}_recnum.parquet')
        )
        # save eddc ppn and episode id
        (df.filter(pl.col('eddc')==1).select(pl.col(['ppn_int','eddc_project_recnum']))
            .collect(streaming=True)
            .write_parquet(f'/mnt/data_volume/apdc/study1/cohort_eddc_group_{i}_recnum.parquet')
        )

        print(df.select(pl.len().over('ppn_int')).collect(streaming=True).describe())
        total_count += df.select(pl.col('ppn_int').n_unique()).collect(streaming=True).item()
    print(f'Total count: {total_count}')

        
if __name__ == '__main__':
    main()
