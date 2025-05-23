import sys
import argparse
import numpy as np
import polars as pl
import polars.selectors as cs

from pathlib import Path
from loguru import logger

# sometimes streaming returns the wrong results
streaming_flag = False

def get_baseline_features(df: pl.DataFrame, lookback_date,
    procedure_dummies=True) -> pl.DataFrame:
    """
    Generates baseline features for a given dataframe.

    Parameters:
        df (polars.DataFrame): The input dataframe containing patient data.

    Returns:
        polars.DataFrame: A new dataframe with baseline features, including numerical and categorical features, 
        as well as procedure code dummies.
    """
    # keep only rows prior to and including index date
    df = df.filter(
        (pl.col('episode_start_date') <= pl.col('index_date')).over('ppn_int')
    )
    # keep rows with a lookback window equal to prediction window
    df = df.filter(
        (pl.col('episode_start_date') >= lookback_date).over('ppn_int')
    )
    df = df.sort(["ppn_int", "episode_start_date"])

    df = df.with_columns(
        pl.concat_list(cs.starts_with('diagnosis')).alias("diagnosis_list")
                            .list.unique().drop_nulls().list.eval(pl.element().filter(pl.element() != ""), 
                            parallel=True)
    )

    #import code
    #code.interact(local=locals())
    #df = df.filter(pl.col('ppn_int')==6724681).write_csv('patient.csv')
    df = df.group_by('ppn_int').agg(
        #pl.all().exclude('procedure_list').last(),
        pl.all().last(),
        procedure_history=(
            pl.concat_list(pl.col('procedure_list')).flatten()
            # keep counts and threshold to binary in the prediction file
            #.unique()
            .drop_nulls()
        #.list.eval(pl.element().filter(pl.element() != "null"))
        ),
        diagnosis_history=pl.col('diagnosis_list').flatten().drop_nulls(),
        num_episodes=pl.len(),
        # naive count, including nested counts
        #lostemp=((pl.col("episode_end_date").cast(pl.Date) - pl.col("episode_start_date").cast(pl.Date))
        #        .dt.total_days()
        #        .fill_null(0)
        #        .clip(lower_bound=0)
        #        + 1 # because same-day ==0
        #    ).sum(),
        # this should be renamed to index_los
        # index variables should be renamed index_foo
        los=((pl.col("finsep").cast(pl.Date) - pl.col("firstadm").cast(pl.Date))
                .dt.total_days()
                .fill_null(0)
                .clip(lower_bound=0)
                + 1 # because same-day ==0
            ).sum(),
        # add sum of "total" column
        prior_cost=pl.col('total').sum(),
        # count number of ed episodes
        ed_episode_count=pl.col('eddc').sum(),
        acute_care=(pl.col('episode_of_care_type') == '1').sum(),
        rehab_care=(pl.col('episode_of_care_type') == '2').sum(),
        mental_care=(pl.col('episode_of_care_type') == 'M').sum(),
    )

    # calculate rurality and sei
    ra = pl.read_excel('SA2_RA_2011_2016.xls')
    sei = pl.read_excel('SA2_ABS_2011_2016.xls')

    ra = dict(zip(ra.get_column('ures9_SA2').to_list(), ra.get_column('ra_code').to_list()))
    sei = dict(zip(sei.get_column('ures9_SA2').to_list(), sei.get_column('Decile').to_list()))
    sa2_col = 'sa2_2016_code'

    # replace sa2 with rurality and sei
    df = df.with_columns(
        pl.col(sa2_col).replace_strict(ra, default=None).alias('rurality'),
        pl.col(sa2_col).replace_strict(sei, default=None).alias('seifa'),
    )
    print(df.select(pl.col('rurality').value_counts()).unnest('rurality').collect())
    print(df.select(pl.col('seifa').value_counts()).unnest('seifa').collect())


    print(df.select(pl.col(['ppn_int', 'episode_start_date', 'procedure_list', 'procedure_history', 'sex'])).head().collect())

    #df = df.with_columns(pl.col('procedure_history').list.join(" "))
    #df = df.group_by('ppn_int').last()
    cat_features = ['sex', 
                    'hospital_type',
                    'ed_status', 
                    'emergency_status_recode', 
                    'episode_of_care_type', 
                    'acute_flag', 
                    'peer_group', 
                    'facility_type', 
                    ] 
    num_features = ['age_recode', 'totlos', 'num_episodes', 'los']

    descriptive_stats = ['diagnosis_history']

    # new baseline variables
    new_cat_features = [
                    'rurality', 
                    'seifa', 
                    ]
    new_num_features = [
                    'prior_cost', 
                    'ed_episode_count',
                    'acute_care',
                    'rehab_care',
                    'mental_care',
                    ]
                

    df = df.select(pl.all().shrink_dtype())
                
    df_base = df.select(pl.col(['ppn_int']+ num_features + cat_features))
    df_base = df_base.collect()
    print(df_base.head())
    df_base = df_base.to_dummies(cat_features)
    print(df_base.head())

    df_new_features = df.select(pl.col(['ppn_int']+ new_num_features + new_cat_features))
    column_names = df_new_features.collect_schema().names()
    rename_features = {c: f'new_{c}' for c in column_names if c != "ppn_int"}
    df_new_features = df_new_features.rename(rename_features).collect()
    df_new_features = df_new_features.to_dummies([f'new_{c}' for c in new_cat_features])

    # convert all procedure codes to counts/dummy variables
    df_codes = df.select(pl.col(['ppn_int', 'procedure_history']))
    df_codes = df_codes.explode('procedure_history')

    # to get procedure counts in lookback history
    df_codes = df_codes.group_by(['ppn_int', 'procedure_history']).agg(
        value=pl.len()
    ).collect(streaming=streaming_flag)

    # procedure dummy variable (yes/no in lookback history)
   # df_codes = df_codes.with_columns(value=pl.lit(1)).collect(streaming=streaming_flag)

    print(df_codes.head(20))
    df_codes = df_codes.pivot(on="procedure_history", index='ppn_int', values='value')
    df_codes = df_codes.fill_null(0)
    df_codes = df_codes.select(pl.all().shrink_dtype())
    # add "procedure" prefix to all procedure columns
    df_codes = df_codes.select(pl.col('ppn_int'), cs.exclude('ppn_int').name.prefix('procedure_'))
    print(df_codes.head(20))

    # add procedure code to other baseline variables
    df_base = df_base.join(df_codes, on='ppn_int', coalesce=True)
    df_base = df_base.join(df_new_features, on='ppn_int', coalesce=True)

    # make all features named "feature_blah"
    rename_features = {c: f'feature_{c}' for c in df_base.columns if c != "ppn_int"}
    df_base = df_base.rename(rename_features)

    # add other variables for descriptive statistics
    df_base = df_base.join(
        df.select(pl.col(['ppn_int']+descriptive_stats)).collect(), 
        on='ppn_int', coalesce=True
    )

    return df_base


def preprocess_dates(df):
    '''
    Make admission and discharge date fields consistent between APDC and EDDC.
    '''
    # convert to date
    df = df.with_columns(
        pl.col(['firstadm','finsep']).str.to_date('%m/%d/%Y'),
        pl.col(['episode_start_date','episode_end_date']).cast(pl.Date),
    )

    # this should all be done in preprocessing of the original files
    df = df.with_columns(
        # this seems to occur only in a few eddc episodes, where end-date is corrupt
        pl.when((pl.col('episode_end_date')-pl.col('episode_start_date')) < 0)
        # set end-date to same as start-date if end-date is corrupt
        .then(pl.col('episode_start_date'))
        .otherwise(pl.col('episode_end_date'))
        .alias('episode_end_date')
    )
    # fill in final separation for ED episodes
    df = df.with_columns(
        pl.col('finsep').fill_null(pl.col('episode_end_date')),
        pl.col('firstadm').fill_null(pl.col('episode_start_date')),
        #pl.col(['firstep','lastep']).fill_null(1),
    )

    # add columns with start and end time for filtering in esgpt
    df = df.with_columns(
            episode_start_datetime=pl.col('episode_start_date').cast(pl.Datetime) 
                    + pl.duration(seconds=pl.col('episode_start_time')),
            episode_end_datetime=pl.col('episode_end_date').cast(pl.Datetime) 
                    + pl.duration(seconds=pl.col('episode_end_time'))
    )

    return df


def label_validation(df: pl.DataFrame, df_result: pl.DataFrame) -> None:
    in_ppn = df.select(pl.col('ppn_int').n_unique()).collect().item()
    out_ppn = df_result.select(pl.col('ppn_int').n_unique()).collect().item()
    assert in_ppn == out_ppn, \
        f"Number of patients after label generation don't match: input={in_ppn}, output={out_ppn}"

    # add a check for duplicates
    n_rows = df_result.select(pl.len()).collect(streaming=streaming_flag).item()
    if n_rows != out_ppn:
        raise ValueError(f"Found duplicate ppn_int entries: {n_rows} rows for {out_ppn} unique ppn_int")


def get_episode_count(df: pl.DataFrame, endpoint: pl.Expr) -> pl.DataFrame:
    """
    This function takes a DataFrame df and a date endpoint as input and returns a new DataFrame
    with the episode count for each patient.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the patient data.
    endpoint : pl.Expr
        Polars expression representing index date and some date offset:
        six months, 1 year, or 2 years.

    Returns
    -------
    pl.DataFrame
        New DataFrame with the episode count for each patient.
    """
    df_result = df.group_by('ppn_int').agg(
        cs.starts_with('index_date').first(),
        episode_count=
            # number of episodes within follow-up window
            (
                (pl.col('episode_start_date') > pl.col('index_date')) &
                (pl.col('episode_start_date') <= endpoint)
            ).sum()
            #+ # add 1 if the 0
            # first because they are all the same value in all the rows
            #(pl.col('died').first() & (pl.col('DEATH_DATE').first() < endpoint)).any().cast(pl.Int8)
    )

    negative_count = df_result.select((pl.col('episode_count')<0).sum()).collect(streaming=streaming_flag).item()
    if negative_count > 0:
        print('error in calculating episode count', negative_count)
        sys.exit(1)

    # validation
    label_validation(df, df_result)

    df_result = df_result.sort('ppn_int')
    return df_result


def calculate_total_days(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates the total number of days for each ppn in a given DataFrame.

    The function takes a DataFrame df as input, which is expected to contain columns 'ppn', 'start_date', and 'end_date'.
    It returns a new DataFrame with a single column 'total_days', where each row corresponds to a unique ppn and the value is the total number of days.

    The calculation involves the following steps:
        1. Sorting the DataFrame by 'ppn' and 'start_date'.
        2. Creating a new column 'group_id' to detect overlapping or adjacent intervals.
        3. Grouping by 'ppn' and 'group_id', and aggregating by taking the minimum 'start_date' and maximum 'end_date'.
        4. Calculating the number of days per merged interval.
        5. Summing the total days for each ppn.

    Parameters:
        df (DataFrame): The input DataFrame containing columns 'ppn', 'episode_start_date', and 'episode_end_date'.

    Returns:
        DataFrame: A new DataFrame with a single column 'total_days', where each row corresponds to a unique ppn.
    
    Todo: This code sometimes returns a list of days for some people. Need to debug.
    """
    # Step 1: Sort by ppn and start_date
    df_sorted = df.sort(by=["ppn_int", "episode_start_date"])

    # Step 2: Create a column that detects where overlaps/adjacent intervals begin a new group
    df_with_group = df_sorted.with_columns(
        (
            (pl.col("episode_start_date") > pl.col("episode_end_date").shift(1).fill_null(pl.date(year=1900,day=1,month=1)) + pl.duration(days=1))
            | (pl.col("ppn_int") != pl.col("ppn_int").shift(1).fill_null(0))
        ).cum_sum().alias("group_id")
    )

    # Step 3: Group by ppn_int and the detected group_id, then aggregate by taking min(episode_start_date) and max(end_date)
    df_merged = df_with_group.group_by(["ppn_int", "group_id"]).agg([
        pl.col("episode_start_date").min().alias("episode_start_date"),
        pl.col("episode_end_date").max().alias("episode_end_date")
    ])

    # Step 4: Calculate the number of days per merged interval
    df_merged = df_merged.with_columns(
        (pl.col("episode_end_date")+pl.duration(days=1) - pl.col("episode_start_date")).alias("days")
    )

    # Step 5: Sum the total days for each ppn_int
    total_days_per_person = df_merged.group_by("ppn_int").agg(
        pl.col("days").sum().dt.total_days().alias("length_of_stay")
    )

    total_days_per_person = total_days_per_person.with_columns(
        pl.col("length_of_stay").map_elements(
            lambda x: x if isinstance(x, int) else x[0], 
            return_dtype=pl.Int64
        )
    )

    return total_days_per_person

def get_length_of_stay(df: pl.DataFrame, endpoint: pl.Expr, maximum: int) -> pl.DataFrame:
    """
    Calculate length of stay (LOS) for each patient.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the patient data.
    endpoint : pl.Expr
        Polars expression representing index date and some date offset:
        six months, 1 year, or 2 years.

    Returns
    -------
    pl.DataFrame
        New DataFrame with the LOS for each patient.
    """
    label = 'length_of_stay'
    # Select episodes that start within the follow-up window
    episodes_within_follow_up = (
        (pl.col("episode_start_date") > pl.col("index_date")) &
        # this ends up including episodes that start in window and end outside window
        (pl.col("episode_start_date") <= endpoint)
    )
    df_los = df.filter(episodes_within_follow_up.over('ppn_int'))

    # Truncate separation time to the end of the follow-up window
    df_los = df_los.with_columns(
        pl.min_horizontal("episode_end_date", endpoint).alias("episode_end_date")
    )

    total_days = calculate_total_days(df_los)

    # check for negative los
    negative_los = total_days.select((pl.col(label)<0).sum()).collect(streaming=streaming_flag).item()
    if negative_los > 0:
        print('error in calculating length of stay', negative_los)
        sys.exit(1)

    df_los = total_days
    
    # select patients with no episodes within follow-up 
    df_los_0 = df.join(df_los, on='ppn_int', how='anti')

    # Set LOS to 0 for patients with no episodes within follow-up window
    df_los_0 = df_los_0.group_by('ppn_int').agg(
        length_of_stay=pl.lit(0),
    )

    # Concatenate the two DataFrames
    df_result = pl.concat([df_los, df_los_0], how='vertical_relaxed')
    # truncate maximum values
    df_result = df_result.with_columns(pl.col(label).clip(upper_bound=maximum))

    # add index dates to label
    df_result = df_result.join(
        df.group_by('ppn_int').agg(cs.starts_with('index_date').first()),
        on='ppn_int'
    )

    # validation
    label_validation(df, df_result)

    df_result = df_result.sort('ppn_int')
    return df_result


def get_cost(df: pl.DataFrame, endpoint: pl.Expr) -> pl.DataFrame:
    '''
    Calculates follow-up cost in three steps:
    1. Cost of episodes entirely contained within follow-up.
    2. Cost of episodes that intersect endpoint (start in follow-up and end after follow-up).
    3. Set cost to 0 for all patients with no episodes within follow-up.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the patient data.
    endpoint : pl.Expr
        Polars expression representing index date and some date offset:
        six months, 1 year, or 2 years.

    Returns
    -------
    pl.DataFrame
        New DataFrame with the cost for each patient.
    '''
    # get all episodes within the follow-up window - differs from los and num of episodes
    # by only including episodes that end within follow-up
    episodes_within_follow_up = ((pl.col('episode_start_date') > pl.col('index_date'))
                                    & (pl.col('episode_end_date') <= endpoint))

    df_cost1 = df.filter(episodes_within_follow_up.over('ppn_int'))

    # count all people with episodes intersecting the endpoint
    truncated_episodes = (
        (pl.col('episode_start_date') > pl.col('index_date'))
        & (pl.col('episode_start_date') < endpoint)
        & (pl.col('episode_end_date') > endpoint)
    )

    df_cost2 = df.filter(truncated_episodes.over('ppn_int'))
    df_cost2 = df_cost2.with_columns(
        (endpoint-pl.col('episode_start_date')).dt.total_days()
        .cast(pl.Int32).alias('valid_days')
    )

    # concatenate complete and truncated episodes
    df_cost = pl.concat([
        df_cost1.select(['ppn_int', 'index_date', 'total']),
        df_cost2.select(
            'ppn_int', 'index_date', 
            (pl.col('total') * pl.col('valid_days') / 
            pl.col('length_of_stay').replace({0:1}).alias('total'))
        )
    ], how='vertical_relaxed')

    # calculate sum cost of complete and truncated episodes
    df_cost = df_cost.group_by('ppn_int').agg(
        cost=pl.col("total").sum().cast(pl.Float32),
    )

    # select patients with no episodes within follow-up ==> cost=0
    df_zero_cost = df.join(df_cost, on='ppn_int', how='anti')
    # set cost to 0
    df_zero_cost = df_zero_cost.group_by('ppn_int').agg(
        pl.lit(0).alias('cost')
    )

    df_result = pl.concat([df_cost, df_zero_cost], how='vertical_relaxed')

    # add index dates to label
    df_result = df_result.join(
        df.group_by('ppn_int').agg(cs.starts_with('index_date').first()),
        on='ppn_int'
    )

    # validation
    label_validation(df, df_result)

    df_result = df_result.sort('ppn_int')
    return df_result


def main():
    """
    Generate dataset, where features consist of embeddings and label
    is time to death. 
    """
    logger.info('Running Pick Prediction Point')
    parser = argparse.ArgumentParser()
    parser.add_argument('split', help='data split: full, tuning or held_out', 
                        choices=['full', 'tuning', 'held_out'], default='tuning')
    parser.add_argument('--features', action='store_true')
    args = parser.parse_args()
    split = args.split

    # configure polars
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(10)
    rng = np.random.default_rng(123)

    parent_path = Path('/mnt/data_volume/apdc/')
    path = parent_path / 'study1/preprocessed/'

    # have to use rbdm, since cod_urf is only available through the end of 2020
    death = pl.scan_parquet(parent_path / 'rbdm_processed.parquet')
    death = death.select(['ppn_int', 'DEATH_DATE'])

    # what is the difference between these two? i.e. if vs else branch?
    if split == 'full':
        # cohort data, including apdc and eddc
        df = pl.scan_parquet(path / 'full_cohort.parquet')
        # this needs to be changed 
        tuning_ppn = pl.scan_parquet(path / 'tuning_ppn.parquet')
        tuning_ppn = tuning_ppn.with_columns(pl.col('subject_id').cast(pl.Int32).alias('ppn_int'))
        # left join since it uses less memory in polars
        df = tuning_ppn.join(df, on='ppn_int', how='left', coalesce=True)
    else:
        df = pl.scan_parquet(path / f'full_cohort_{split}.parquet')

    '''
    sizes = pl.DataFrame({'column':df.columns, 
                'size_bytes':[df.select(c).estimated_size() for c in df.columns]})
    
    sizes = sizes.sort('size_bytes', descending=True)
    sizes = sizes.with_columns((pl.col('size_bytes').cast(pl.Float32)/(1024*1024)).alias('mb'))
    print(sizes)
    '''

    # drop expensive columns to reduce memory usage during computations
    #df = df.drop(['procedure_list', 'ppn'])
    df = df.drop(['ppn'])

    df = preprocess_dates(df)
    #print('df samples', df.select(pl.col('ppn_int').n_unique()).collect())
    #df = df.with_columns(pl.col('finsep').backward_fill().over('ppn_int'))

    # read imputed cost data
    abf_dnr = pl.scan_parquet(path / f'full_cohort_{split}_cost_imputed_truncated.parquet')
    abf_dnr = abf_dnr.with_columns(
        pl.coalesce(['apdc_project_recnum', 'eddc_project_recnum']).alias('recnum')
    )
    abf_dnr = abf_dnr.select(['recnum', 'total'])
    #print('abf samples', abf_dnr.select(pl.col('ppn_int').n_unique()).collect())

    #df = df.head(10000).collect(streaming=streaming_flag)

    # select only hospital episodes, the discharge episode of hospital stays
    # this leads to circumstances where nested episodes are selected because they are labeled as "last episode",
    # even though they do not have the last discharge date
    #apdc = df.filter((pl.col('apdc')==1) & (pl.col('lastep')==1))
    # apdc episodes only
    apdc = df.filter(pl.col('apdc')==1)
    # select episodes whose end date is equal to finsep date
    apdc = df.filter(pl.col('episode_end_date')==pl.col('finsep'))
    # join apdc and death
    #df = df.join(death.collect(), on='ppn_int', how='left', coalesce=True)
    apdc = apdc.join(death, on='ppn_int', how='left', coalesce=True)
    apdc = apdc.with_columns(pl.col('DEATH_DATE').is_not_null().alias('died'))
    print(apdc.select(pl.col('ppn_int').n_unique()).collect())
    # exclude hospital stays where patient dies from pool of eligible episodes for index date
    apdc = apdc.filter(
        # eligible episodes include all episodes for people who didn't die
        # or episodes that ended before the death date (exclude episodes where patient dies)
        ~pl.col('died') | (pl.col('episode_end_date') < pl.col('DEATH_DATE'))
    )
    print(apdc.select(pl.col('ppn_int').n_unique()).collect())
    print(apdc.select((pl.col('episode_end_date').cast(pl.Date) >= pl.col('DEATH_DATE')).sum()).collect())

    # have to sort here, otherwise it leads to different row orders in different runs
    apdc = apdc.sort(['ppn_int','episode_start_date'])
    # add row index per patient to randomly select one
    apdc = apdc.with_columns(pl.int_range(pl.len()).over('ppn_int').alias('row_index'))

    num_rows = apdc.group_by('ppn_int').len().collect()
    rand_index = rng.integers(0, num_rows.get_column('len').to_numpy())
    num_rows = num_rows.with_columns(index_row=rand_index)
    apdc = apdc.join(num_rows.select(pl.col(['ppn_int', 'index_row'])).lazy(), 
                    on='ppn_int', coalesce=True)
    #print(num_rows)
    #print(apdc.select((pl.col('row_index')==pl.col('index_row')).sum()).collect())
    #print(apdc.select(pl.col('ppn_int').n_unique()).collect())
    index_date_episodes = apdc.filter(pl.col('row_index')==pl.col('index_row'))
    n_unique = index_date_episodes.select(pl.col('ppn_int').n_unique()).collect().item()
    n_rows = index_date_episodes.select(pl.len()).collect().item()
    # adding index date+time as a column to be used for finding index point in esgpt data
    if n_unique != n_rows:
        print(f'Multiple index rows for one person: ppn {n_unique} and rows {n_rows}')
        raise ValueError("Rewrite program to multiple index dates returned for one patient")

    # randomly select one episode per patient
    #index_date_episodes = apdc.filter(pl.col('row_index').shuffle().over('ppn_int') < 1)

    # right now this doesn't matter because I have changed code to use discharge date
    # using finsep chooses the stay discharge
    # for episode_end_date, have to make sure that time has been added to it
    discharge_col = 'finsep' #'episode_end_date' # 'finsep'

    index_date = index_date_episodes.with_columns(
        pl.col(discharge_col).alias('index_date'),
        pl.col('episode_end_datetime').alias('index_datetime')
    )
    index_date = index_date.select(pl.col(['ppn_int', 'index_date', 'index_datetime', 'DEATH_DATE', 'died']))
    # adds index date and death as columns
    df = df.join(index_date, on='ppn_int', coalesce=True)

    # add cost data
    #print(df.select(pl.len()).collect())
    df = df.with_columns(
        pl.coalesce(['apdc_project_recnum', 'eddc_project_recnum']).alias('recnum')
    )
    print('after coalesce', df.select(pl.len()).collect())
    # add cost data to episodes
    # here I get an increase in numbers, that means that some episodes show up twice
    # in abf-dnr
    # need to drop duplicates
    #print('df samples', df.select(pl.col('ppn_int').n_unique()).collect())
    df = df.join(abf_dnr, on='recnum', coalesce=True)
    #print('last', df.select(pl.len()).collect())
    #print('df samples', df.select(pl.col('ppn_int').n_unique()).collect())
    #return

    # calculates number of days between index date and discharge of last available episode ((why?))
    #df = df.with_columns((pl.col(discharge_col).max()-pl.col('index_date')).dt.total_days().over('ppn_int').alias('last_episode_diff'))
    # calculates number of days between index date and last available follow-up
    df = df.with_columns((pl.date(2021, month=6, day=30)-pl.col('index_date')).dt.total_days().over('ppn_int').alias('end_followup_diff'))
    print(df.select((pl.col('end_followup_diff')<0).sum()).collect())

    # sets death date as end-of-followup for those who died
    df = df.with_columns(
        pl.when(pl.col('died'))
        .then((pl.col('DEATH_DATE') - pl.col('index_date')).dt.total_days())
        .otherwise(pl.col('end_followup_diff'))
        .alias('end_followup_diff')
    )
    # I don't understand how I get negative end of follow-up here
    # it also changes during different runs, which should also not be possible based 
    # on setting random seed
    print(df.select((pl.col('end_followup_diff')<0).sum()).collect())
    #(df.filter(pl.col('end_followup_diff')<0).select(
    #    pl.col(['ppn_int', 'index_date', 'episode_start_date', 
    #    'episode_end_date','DEATH_DATE', 'died', 'apdc', 'eddc']))
    #.collect().write_csv('test.csv'))
    #return

    #import pdb
    #pdb.set_trace()
    #return
    #import code
    #code.interact(local=locals())

    # set three flags to mark inclusion in the 6months/1 year/2years sets based 
    # on available follow-up
    df = df.with_columns(
        # if more than 6 months of follow-up, or if died in the six months after index date
        ((pl.col('end_followup_diff') > 180) | pl.col('died')).alias('six_months'),
        # if more than 1 year of follow-up, or if died in the near after index date
        ((pl.col('end_followup_diff') > 365) | pl.col('died')).alias('one_year'),
        ((pl.col('end_followup_diff') > 730) | pl.col('died')).alias('two_year'),
    )

    # I can't figure out why some patients still have negative episodes here
    #print(df.select((pl.col('end_followup_diff')<0).sum()).collect())
    #(df.filter(pl.col('end_followup_diff')<0).select(
    #    pl.col(['ppn_int', 'index_date', 'episode_start_date', 
    #    'episode_end_date','firstadm', 'finsep', 'DEATH_DATE', 'died', 'apdc', 'eddc']))
    #.collect().write_csv('test.csv'))
    #return

    # remove all people with 0 or negative follow-up
    # todo: debug this, it should not be possible
    #df = df.filter(pl.col('end_followup_diff') > 0)

    #---------------------------------------------
    result_path = path / 'tasks'
    windows = {
        'six_months': '6mo', 
        'one_year': '1y', 
        'two_year': '2y', 
    }
    days_max = {
        '6mo': 180,
        '1y': 365,
        '2y': 365*2,
    }

    df = df.sort(["ppn_int", "episode_start_date"])
    # make sure everyone has the two years of follow-up, which ensures everyone is aligned to two-year dataset
    df_window = df.filter(pl.col('two_year'))
    #import code
    #code.interact(local=locals())
    lookback_point = pl.col('index_date').dt.offset_by('-2y')
    baseline_features = get_baseline_features(df_window, lookback_point)
    baseline_features.write_parquet(result_path / f'baseline_features_strong_two_year.parquet')
    for window, offset in windows.items():
        #df_window = df.filter(pl.col(window))
        endpoint = pl.col('index_date').dt.offset_by(offset)

        # cost
        logger.info(f'Calculating cost for {window}')
        df_cost = get_cost(df_window, endpoint=endpoint)
        df_cost.collect(streaming=streaming_flag).write_parquet(result_path / f'cost_{window}.parquet')

        # length of stay
        logger.info(f'Calculating and saving total length of stay for {window}')
        df_los = get_length_of_stay(df_window, endpoint=endpoint, maximum=days_max[offset])
        df_los.collect(streaming=streaming_flag).write_parquet(result_path / f'length_of_stay_{window}.parquet')

        # episode count
        logger.info(f'Calculating and saving episode count for {window}')
        df_ep_count = get_episode_count(df_window, endpoint=endpoint)
        df_ep_count.collect(streaming=streaming_flag).write_parquet(result_path / f'episode_count_{window}.parquet')




if __name__ == "__main__":
    main()
