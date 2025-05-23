import argparse
import gc
import os
import sys
import time
from pathlib import Path
import polars as pl
import pandas as pd
from loguru import logger

import matplotlib.pyplot as plt
import miceforest as mf

# streaming engine of polars 1.28 gives weird results and causes errors
#streaming = {'engine': 'streaming'}
# polars should be pinned to 1.23.0
streaming_flag = False

OTHER_CATEGORY_THRESHOLD = 50
#OTHER_CATEGORY_THRESHOLD = 500
imputation_params = {'verbose':True,
    #lgbm params
    #'min_data_in_leaf':20, 
    'max_depth':6, 
    'num_iterations': 20,
    'num_leaves': 20,
    #'learning_rate': 0.5,
    'verbosity': 1,
}
CPI_YEAR = 2021
SEED = 123
MICE_ITERATIONS = 2 #3

def conditional_set_memory_limit(value):
    def wrapper(func):
        if os.name == 'nt':
            from kill import set_memory_limit
            return set_memory_limit(value)(func)
        return func
    return wrapper

def group_other_categories(df, categories, threshold=20):
    replace = {}
    # identify categories below threshold
    for c in categories:
        counts = df.select(pl.col(c).value_counts(sort=True)).collect(streaming=streaming_flag).unnest(c)
        counts = counts.filter(pl.col('count') < threshold).get_column(c).to_list()
        replace[c] = {c:'other' for c in counts}

    # replace them with "other"
    df = df.with_columns(
        *[pl.col(c).replace(replace[c]) for c in categories]
    )
    return df


def preprocess_data(data_name, df, categories):
    if data_name == 'apdc':
        df = df.with_columns(
            pl.col('diagnosis_codep').str.replace(r'\..*$', ''),
            pl.col('indigenous_status').forward_fill().backward_fill().fill_null('9')
        )
    else:
        # eddc
        df = df.with_columns(
            pl.col('ed_diagnosis_code').str.replace(r'\..*$', ''),
            pl.col('indigenous_status').forward_fill().backward_fill().fill_null('9')
        )
        df = df.with_columns(
            pl.when(pl.col('ed_diagnosis_code').is_null())
            .then(pl.col('ed_diagnosis_code_sct'))
            .otherwise(pl.col('ed_diagnosis_code'))
            .alias('ed_diagnosis_code')
        )

    df = group_other_categories(df, categories, threshold=OTHER_CATEGORY_THRESHOLD)
    return df


def get_cpi():
    df = pd.read_csv('cpi_june.csv')
    ref = df.loc[df['year']==CPI_YEAR].iloc[0]
    df['cpi'] = ref['cpi'] / df['cpi']
    cpi = df.set_index('year')['cpi'].to_dict()
    return cpi


def impute_data(data: pd.DataFrame, variable_schema: dict, impute_col: str, fout:str,
    chunk_size: int = 100000
) -> pd.DataFrame:
    """
    """
    logger.info('Imputing data')
    start = time.time()
    n_rows = len(data)
    n_chunks = (n_rows + chunk_size - 1) // chunk_size
    if n_rows < chunk_size:
        chunk_size = n_rows
        n_chunks = 1

    results = []

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i+1) * chunk_size, n_rows)
        logger.info(f'Imputing chunk {i+1} of size {end_idx-start_idx}')
        chunk = data.iloc[start_idx:end_idx].copy().reset_index(drop=True)

        kernel = mf.ImputationKernel(
            chunk,
            random_state=SEED,
            variable_schema=variable_schema,
            #mean_match_candidates=3,
            #data_subset=20000,
            mean_match_strategy="fast"
        )
        kernel.mice(MICE_ITERATIONS,
                #variable_parameters={impute_col: {'objective': 'poisson'}},
                variable_parameters={impute_col: {'objective': 'regression'}},
                **imputation_params
                )
        complete_data = kernel.complete_data(dataset=0)
        impute_model = kernel.get_model(dataset=0, variable=impute_col)
        print(impute_model.params)
        #kernel.plot_mean_convergence()
        #plt.savefig(fout)
        results.append(complete_data)

    complete_data = pd.concat(results, ignore_index=True)
    end = time.time()
    logger.info(f'Execution time {end-start} seconds')
    print(complete_data[impute_col].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.90]))
    return complete_data


def get_cost_data(source, recnum, cost_col):
    # read apdc cost data
    abf_dnr = pl.scan_parquet(source)
    abf_dnr = abf_dnr.rename(lambda c: c.lower())
    abf_dnr = abf_dnr.with_columns(
        pl.col(cost_col).cast(pl.Float32)
    )

    #print(abf_dnr.collect_schema().names())
    #sys.exit()
    #print(abf_dnr.select(pl.col(cost_col).is_null().sum()).collect())
    #abf_dnr = abf_dnr.with_columns(
    #    pl.col(cost_col)
    #    .cast(pl.Float32)
    #    .clip(lower_bound=0)
    #    .replace({0:None})
    #)
    #print(abf_dnr.select(pl.col(cost_col).is_null().sum()).collect())
    print(source)
    print('null cost')
    print(abf_dnr.select(pl.col(cost_col).is_null().sum()).collect())

    print('negative cost')
    print(abf_dnr.select((pl.col(cost_col) < 0).sum()).collect())

    abf_dnr = abf_dnr.select([recnum, cost_col])
    return abf_dnr

def apdc_eddc_impute(data_name, df, project_path, chunk_size=100000):
    if data_name == 'apdc':
        recnum = 'apdc_project_recnum'
        categories = [
            'sex', 'block_nump', 'diagnosis_codep', 'indigenous_status', 'emergency_status_recode'
            ]
    else:
        recnum = 'eddc_project_recnum'
        categories = [
            'sex', 'ed_diagnosis_code', 'indigenous_status', 'triage_category', 'ed_visit_type'
        ]

    cost_col = 'total'
    impute_cols = ['age_recode', 'episode_length_of_stay'] + categories

    df = preprocess_data(data_name, df, categories=categories)
    df = df.with_columns(pl.col(categories).cast(pl.Categorical))

    # cost data
    abf_dnr = get_cost_data(project_path / f'{data_name}*dnr*.parquet', recnum, cost_col)

    # add episode date to episodes with cost
    abf_dnr = abf_dnr.join(
        df.select(pl.col([recnum, 'episode_start_date'])), 
        on=recnum, how='inner', coalesce=True
    )

    print('Distribution before cpi adjustment')
    percentile = [p/100 for p in [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]]
    print(abf_dnr.select(pl.col(cost_col)).describe(percentiles=percentile))

    # adjust cost for inflation
    cpi = get_cpi()
    abf_dnr = abf_dnr.with_columns(
        pl.col('episode_start_date').dt.year().alias('year')
    )
    abf_dnr = abf_dnr.with_columns(
        pl.col('year').cast(pl.Float32).replace(cpi).alias('cpi')
    )
    abf_dnr = abf_dnr.with_columns(
        (pl.col(cost_col)*pl.col('cpi')).alias(cost_col)
    )

    # show cost distribution stats
    print('Distribution after cpi adjustment')
    percentile = [p/100 for p in [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]]
    print(abf_dnr.select(pl.col(cost_col)).describe(percentiles=percentile))

    percentile = [p/100 for p in [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]]
    high = abf_dnr.filter(pl.col(cost_col) > pl.col(cost_col).quantile(.975))
    print(high.select(pl.col(cost_col)).describe(percentiles=percentile))

    lower_limit = abf_dnr.select(pl.col(cost_col).quantile(.025)).collect().item()
    upper_limit = abf_dnr.select(pl.col(cost_col).quantile(.975)).collect().item()

    # drop cost episodes with cost below lower limit and above upper limit
    #abf_dnr = abf_dnr.filter(pl.col(cost_col) > lower_limit)
    #abf_dnr = abf_dnr.filter(pl.col(cost_col) < upper_limit)

    # cap cost to 2.5 and .975 percentiles 
    abf_dnr = abf_dnr.with_columns(
        pl.col(cost_col).clip(lower_bound=lower_limit, upper_bound=upper_limit)
    )

    # keep only relevant columns to reduce memory during collect
    df = df.select(pl.col([recnum, 'ppn_int'] + impute_cols))

    # add cost data to episodes
    df = df.join(
        abf_dnr.select(pl.col([recnum, cost_col])), 
        on=recnum, how='left', coalesce=True
    )

    logger.info(f'Collecting {data_name} data set')
    #df = df.head(200000).collect(streaming=streaming_flag)
    df = df.collect(streaming=streaming_flag)

    variable_schema = { cost_col: impute_cols, }
    data = df.drop(['ppn_int', recnum]).to_pandas()
    complete_data = impute_data(
        data, variable_schema, cost_col, f'{data_name}_convergence.png',
        chunk_size=chunk_size,
    )

    df = df.select(pl.col(['ppn_int', recnum]))
    df = df.with_columns(total=complete_data.total.values)
    return df


@conditional_set_memory_limit(29)
def main():
    '''
    Imputes cost for APDC and EDDC. 
    APDC and EDDC are imputed separately, as they have different fields.
    This current version assumes entire dataset fits in memory.
    '''
    logger.info('Running impute cost')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='input file')
    parser.add_argument('--output', type=str, help='output file')
    parser.add_argument('--abf-dnr-path', type=str, help='file with cost')
    args = parser.parse_args()

    pl.Config.set_tbl_rows(100)

    # read cohort split
    # select only apdc episodes
    # join with cost episodes
    # impute
    # save to file

    # read data
    df = pl.scan_parquet(args.input_file)
    #df = df.head(200000)

    abf_dnr_path = Path(args.abf_dnr_path)

    #apdc = df.filter(pl.col('apdc')==1).select(pl.col('diagnosis_codep').n_unique())
    #eddc = df.filter(pl.col('eddc')==1).select(pl.col('ed_diagnosis_code').n_unique())
    #logger.info(f"apdc {apdc.collect().item()},eddc{eddc.collect().item()}")
    #return

    # repeat for eddc
    logger.info('Imputing EDDC')
    eddc = apdc_eddc_impute(
        'eddc', 
        df.filter(pl.col('eddc')==1), 
        abf_dnr_path,
        # higher chunk_size for EDDC makes things incredibly slow
        chunk_size=10000,
    )

    # impute apdc
    logger.info('Imputing APDC')
    apdc = apdc_eddc_impute(
        'apdc', 
        df.filter(pl.col('apdc')==1), 
        abf_dnr_path,
        chunk_size=500000
    )
    apdc.write_parquet('__tmp__apdc_cost_impute.parquet')
    eddc.write_parquet('__tmp__eddc_cost_impute.parquet')
    del apdc
    del eddc

    apdc = pl.scan_parquet('__tmp__apdc_cost_impute.parquet')
    eddc = pl.scan_parquet('__tmp__eddc_cost_impute.parquet')

    logger.info('Merging imputed APDC and EDDC and saving')
    df_cost = pl.concat([apdc, eddc], how='diagonal')

    df_cost = df_cost.with_columns(
        pl.coalesce(['apdc_project_recnum', 'eddc_project_recnum']).alias('recnum')
    )

    df = df.with_columns(
        pl.coalesce(['apdc_project_recnum', 'eddc_project_recnum']).alias('recnum')
    )

    df = df.join(df_cost, on='recnum')
    df.sink_parquet(args.output)

    os.remove('__tmp__apdc_cost_impute.parquet')
    os.remove('__tmp__eddc_cost_impute.parquet')


if __name__ == '__main__':
    main()
