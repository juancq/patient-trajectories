# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:50:27 2023

@author: juanqa
"""
import polars as pl
from tqdm import tqdm


def filter_cohort():
    """
    The main function processes patient data from the APDC and EDDC datasets, 
    filtering by inclusion/exclusion criteria and saving the results to parquet files.

    The inclusin/exclusion criteria are defined in the `inclusion_exclusion_criteria.py`.

    It takes no parameters and returns no values, but instead writes the processed 
    data to files in the specified destination path.

    The function processes the data in groups to avoid memory issues.
    """
    src_path = '/mnt/data_volume/apdc/str_group_files'
    dest_path = '/mnt/data_volume/apdc/study1'
    
    num_groups = 50
    total_count = 0
    # used to count the number of patients in each group to ensure no patients are dropped
    ppn_set = set()
    # process each patient group, processing by group necessary to avoid memory issues
    for i in tqdm(range(num_groups)):

        # read row records for each group that meet inclusion/exclusion criteria
        apdc_recnum = pl.scan_parquet(f'{dest_path}/cohort_apdc_group_{i}_recnum.parquet')
        eddc_recnum = pl.scan_parquet(f'{dest_path}/cohort_eddc_group_{i}_recnum.parquet')

        # read apdc and eddc data
        apdc = pl.scan_parquet(f'{src_path}/apdc_group_{i}_str.parquet')
        ed = pl.scan_parquet(f'{src_path}/eddc_group_{i}_str.parquet')

        # keep only rows that meet inclusion/exclusion criteria
        apdc = apdc.join(apdc_recnum.select('apdc_project_recnum'), on='apdc_project_recnum', coalesce=True)
        ed = ed.join(eddc_recnum.select('eddc_project_recnum'), on='eddc_project_recnum', coalesce=True)

        # save to file
        apdc.collect(streaming=True).write_parquet(f'{dest_path}/cohort_apdc_group_{i}_data.parquet')
        ed.collect(streaming=True).write_parquet(f'{dest_path}/cohort_eddc_group_{i}_data.parquet')

        # counting for debugging purposes
        df = pl.concat([apdc_recnum, eddc_recnum], how='diagonal')
        total_count += df.select(pl.col('ppn_int').n_unique()).collect(streaming=True).item()
        
    print(f'total_count count: {total_count}')

        
if __name__ == '__main__':
    filter_cohort()