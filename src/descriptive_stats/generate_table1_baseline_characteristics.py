from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
from scipy import stats
from typing import List, Dict, Union
import utils.predict_utils as ut
from utils.kill import conditional_set_memory_limit
from tableone import TableOne


@conditional_set_memory_limit(24)
def main():
    # Create example dataset with three unrelated outcomes
    np.random.seed(42)
    '''
    n = 1000
    
    df = pd.DataFrame({
        # Outcomes
        'outcome1': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'outcome2': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'outcome3': np.random.choice([0, 1], n, p=[0.5, 0.5]),
        
        # Baseline characteristics
        'age': np.random.normal(50, 10, n),
        'weight': np.random.normal(70, 15, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'smoking_status': np.random.choice(['Smoker', 'Non-Smoker'], n),
        'comorbidity': np.random.choice(['Yes', 'No'], n)
    })
    '''

    config = ut.load_config('config/predict_config.yaml')
    label_path = Path(config.label_path_nt)
    window = 'two_year'
    df = pl.scan_parquet(label_path / f'baseline_features_strong_two_year.parquet')
    for task in config.tasks:
        df_label = pl.scan_parquet(label_path / f'{task}_{window}.parquet')
        df_label = ut.threshold_label(df_label.collect(), task, True)
        df_label = df_label.with_columns(pl.col(task).alias(f'outcome_{task}'))
        df_label = df_label.lazy()
        df = df.join(df_label.select(pl.col(['ppn_int', f'outcome_{task}'])), on='ppn_int')

    df = df.with_columns(
        (pl.col('feature_num_episodes') >=5).cast(pl.Int8).alias('num_episodes>=5'),
        (pl.col('feature_num_episodes').is_between(2, 4)).cast(pl.Int8).alias('num_episodes=2-4'),
        (pl.col('feature_num_episodes') ==1).cast(pl.Int8).alias('num_episodes=1'),
        (pl.col('diagnosis_history').list.eval(pl.element().str.contains('I[1-7]')))
        .list.any().cast(pl.Int8).alias('cvd')
    )

    df = df.with_columns(
        pl.when(pl.col('feature_num_episodes')==1).then(0)
        .when(pl.col('feature_num_episodes').is_between(2,4)).then(1)
        .otherwise(2)
        .alias('episode_group')
    )

    df = df.collect().to_pandas()
    df.rename(columns=lambda x: x.replace('feature_new_', ''), inplace=True)
    df.rename(columns=lambda x: x.replace('feature_', ''), inplace=True)
    #df.isna().sum().to_csv('hello.csv')

    outcomes = [f'outcome_{task}' for task in config.tasks]

    categorical_vars = [
        'sex_2', 'hospital_type_true','num_episodes>=5',
        'cvd',
        'episode_group'
        #'num_episodes=1', 'num_episodes=2-4', 'cvd',
    ]
    no_decimals = [
        'age_recode', 'num_episodes', 'los', 'ed_episode_count',
        'acute_care',
        'rehab_care',
        'mental_care',
    ]

    continuous_vars= no_decimals + ['prior_cost',]

    # set decimal places for these variables to 0
    decimals = {c: 0 for c in no_decimals}

    # limit categorical variable to a single row, it shows the most frequent category
    limit = {c: 1 for c in categorical_vars if c != 'episode_group'}
    
    for i, outcome in enumerate(outcomes):
        overall = i == 0
        baseline_table = TableOne(df, categorical=categorical_vars,
            continuous=continuous_vars, nonnormal=continuous_vars,
            groupby=outcome,
            limit=limit, decimals=decimals,
            pval=True,
            missing=False,
            overall=overall,
            )
        print(baseline_table)
        baseline_table.to_csv(f'table1_{outcome}.csv')
    

if __name__ == "__main__":
    main()
