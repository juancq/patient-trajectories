import pickle
import os
import sys
import argparse
import numpy as np
import polars as pl
import polars.selectors as cs

from pathlib import Path
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder

# sometimes streaming returns the wrong results
#streaming = {'engine': 'streaming'}
streaming_flag = False


def save_feature_metadata(metadata, filepath):
    """
    Save feature metadata (like unique procedure codes) to a file.
    
    Parameters:
        metadata (dict): Dictionary containing metadata about features
        filepath (str or Path): Path to save the metadata
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(metadata, f)

def load_feature_metadata(filepath):
    """
    Load feature metadata from a file.
    
    Parameters:
        filepath (str or Path): Path to the metadata file
        
    Returns:
        dict: Dictionary containing metadata about features
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def get_baseline_features(df: pl.DataFrame, 
                          lookback_date,
                          procedure_dummies=True,
                          metadata_path=None,
                          is_train=False) -> pl.DataFrame:

    """
    Generates baseline features for a given dataframe ensuring consistency between train and test sets.

    Parameters:
        df (polars.DataFrame): The input dataframe containing patient data.
        lookback_date: The lookback date for filtering data
        procedure_dummies (bool): Whether to create procedure code dummies
        metadata_path (str or Path): Path to save/load feature metadata
        is_train (bool): Whether this is the training set (to save metadata) or not (to load metadata)

    Returns:
        polars.DataFrame: A new dataframe with baseline features, including numerical and categorical features, 
        as well as procedure code dummies.
    """
    train_test_label = 'train' if is_train else 'test'
    metadata_path = Path(metadata_path)
    # keep only rows prior to and including index date
    print(df.select(pl.col('ppn_int').n_unique()).collect())
    #df2 = df.filter(
    #    (pl.col('episode_start_date') <= pl.col('index_date')).over('ppn_int')
    #)
    #print(df2.select(pl.col('ppn_int').n_unique()).collect())
    df = df.filter(
        (pl.col('episode_start_date') <= pl.col('index_date'))
    )
    print(df.select(pl.col('ppn_int').n_unique()).collect())
    #import code
    #code.interact(local=locals())
    # keep rows within lookback window
    df = df.filter(
        #(pl.col('episode_start_date') >= lookback_date).over('ppn_int')
        (pl.col('episode_start_date') >= lookback_date)
    )
    print(df.head().collect())

    #df = df.with_columns((pl.col('episode_start_date') >= lookback_date).over('ppn_int').alias('flag'))
    #df.head().collect()
    df = df.sort(["ppn_int", "episode_start_date"])
    print(df.head().collect())

    df = df.with_columns(
        pl.concat_list(cs.starts_with('diagnosis')).alias("diagnosis_list")
                            .list.unique().drop_nulls().list.eval(pl.element().filter(pl.element() != ""), 
                            parallel=True)
    )

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
    logger.info('Rurality summary')
    logger.info(df.select(pl.col('rurality').value_counts()).unnest('rurality').collect())
    logger.info('Socioeconomic summary')
    logger.info(df.select(pl.col('seifa').value_counts()).unnest('seifa').collect())

    print(df.select(pl.col(['ppn_int', 'episode_start_date', 'procedure_list', 'procedure_history', 'sex'])).head().collect())

    cat_features = ['sex', 
                    'hospital_type',
                    'ed_status', 
                    'emergency_status_recode', 
                    'episode_of_care_type', 
                    'acute_flag', 
                    'peer_group', 
                    'facility_type', 
                    'rurality', 
                    'seifa', 
                    ] 
    num_features = ['age_recode', 'totlos', 'num_episodes', 'los',
                    'prior_cost', 
                    'ed_episode_count',
                    'acute_care',
                    'rehab_care',
                    'mental_care',
    ]

    descriptive_stats = ['diagnosis_history']

    df = df.select(pl.all().shrink_dtype())

    # keep only the first 3 digits of ICD diagnosis codes
    df = df.with_columns(
        pl.col('diagnosis_history').list.eval(pl.element().str.split('.').list.get(0))
    )

    metadata_filename = metadata_path / 'temporal_feature_encoders.pkl'
                
    # Initialize metadata dictionary if this is the training set
    if is_train and metadata_path is not None:
        logger.info('Building metadata for train')
        metadata = {}
        
        # Extract and save procedure codes from training set
        procedure_series = df.select('procedure_history').collect().get_column('procedure_history').to_list()

        logger.info('proc vectorizer')
        #vectorizer = CountVectorizer(token_pattern=r"[^\s]+")
        proc_vectorizer = MultiLabelBinarizer()
        proc_vectorizer.fit(procedure_series)
        metadata['procedures'] = proc_vectorizer

        del procedure_series

        logger.info('diag vectorizer')
        df_diag_train = df.select('diagnosis_history')
        df_diag_train = df_diag_train.with_columns(
            pl.col('diagnosis_history').list.eval(pl.element().str.split('.').list.get(0))
        )
        diagnosis_series = df_diag_train.select('diagnosis_history').collect().get_column('diagnosis_history').to_list()

        #diagosis_vectorizer = CountVectorizer(token_pattern=r"[^\s]+")
        diagosis_vectorizer = MultiLabelBinarizer()
        diagosis_vectorizer.fit(diagnosis_series)
        metadata['diagnosis'] = diagosis_vectorizer
        del diagnosis_series

        category_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        df_categorical = df.select(cat_features).collect()
        category_encoder.fit(df_categorical)
        metadata['category'] = category_encoder
        del df_categorical
        
        # Save metadata
        logger.info('Saving train metadata')
        save_feature_metadata(metadata, metadata_filename)
    
    # Load metadata if this is not the training set
    elif metadata_path is not None and os.path.exists(metadata_path):
        logger.info('Loading train metadata')
        metadata = load_feature_metadata(metadata_filename)
    else:
        metadata = None

    # numerical features
    logger.info('Saving numerical features')
    df_num = df.select(pl.col(['ppn_int'] + num_features))
    df_num.collect().write_parquet(metadata_path / f'{train_test_label}_numerical_features.parquet')

    # categorical features
    logger.info('Saving categorical features')
    df_cat = df.select(pl.col(['ppn_int'] + cat_features)).collect()

    categorical_encoder = metadata['category']
    cat_features = categorical_encoder.transform(df_cat.select(cat_features))
    #feature_names = [f'procedure_{name}' for name in vectorizer.get_feature_names_out()]
    feature_names = categorical_encoder.get_feature_names_out()

    logger.info(cat_features.shape)
    logger.info(cat_features)

    df_cat_encoded = pl.DataFrame(
        cat_features,
        schema=feature_names.tolist()
    ).with_columns(df_cat.get_column('ppn_int'))
    df_cat_encoded.write_parquet(metadata_path / f'{train_test_label}_categorical_features.parquet')
    del df_cat
    del df_cat_encoded
    del cat_features

    # procedure features
    logger.info('Saving procedure features')
    df_procedures = df.select(pl.col(['ppn_int', 'procedure_history'])).collect()
    procedure_encoder = metadata['procedures']
    procedure_features = procedure_encoder.transform(df_procedures.get_column('procedure_history'))
    #feature_names = [f'procedure_{name}' for name in procedure_encoder.get_feature_names_out()]
    feature_names = [f'procedure_{name}' for name in procedure_encoder.classes_]

    logger.info(procedure_features.shape)
    logger.info(procedure_features)

    df_procedures_encoded = pl.DataFrame(
        procedure_features,
        schema=feature_names,
    ).with_columns(df_procedures.get_column('ppn_int'))
    df_procedures_encoded.write_parquet(metadata_path / f'{train_test_label}_procedure_features.parquet')
    del df_procedures
    del df_procedures_encoded
    del procedure_features

    # diagnosis features
    logger.info('Saving diagnosis features')
    df_diagnosis = df.select(pl.col(['ppn_int', 'diagnosis_history'])).collect()
    diagnosis_encoder = metadata['diagnosis']
    diagnosis_features = diagnosis_encoder.transform(df_diagnosis.get_column('diagnosis_history'))
    #feature_names = [f'diagnosis_{name}' for name in diagnosis_encoder.get_feature_names_out()]
    feature_names = [f'diagnosis_{name}' for name in diagnosis_encoder.classes_]

    logger.info(diagnosis_features.shape)
    logger.info(diagnosis_features)

    df_diagnosis_encoded = pl.DataFrame(
        diagnosis_features,
        schema=feature_names,
    ).with_columns(df_diagnosis.get_column('ppn_int'))
    df_diagnosis_encoded.write_parquet(metadata_path / f'{train_test_label}_diagnosis_features.parquet')
    del df_diagnosis
    del df_diagnosis_encoded
    del diagnosis_features


    logger.info('joining dataframes')

    # Add procedure code to other baseline variables
    df_num = pl.scan_parquet(metadata_path / f'{train_test_label}_numerical_features.parquet')
    df_cat = pl.scan_parquet(metadata_path / f'{train_test_label}_categorical_features.parquet')
    df_procedures = pl.scan_parquet(metadata_path / f'{train_test_label}_procedure_features.parquet')
    df_diagnosis = pl.scan_parquet(metadata_path / f'{train_test_label}_diagnosis_features.parquet')

    df_base = df.select(pl.col('ppn_int'))
    df_base = df_base.join(df_num, on='ppn_int', coalesce=True)
    df_base = df_base.join(df_cat, on='ppn_int', coalesce=True)
    df_base = df_base.join(df_procedures, on='ppn_int', coalesce=True)
    df_base = df_base.join(df_diagnosis, on='ppn_int', coalesce=True)

    # Make all features named "feature_blah"
    rename_features = {c: f'feature_{c}' for c in df_base.collect_schema().names() if c != "ppn_int"}
    df_base = df_base.rename(rename_features)

    # Add other variables for descriptive statistics
    df_base = df_base.join(
        df.select(pl.col(['ppn_int'] + descriptive_stats)),#.collect(), 
        on='ppn_int', coalesce=True
    )
    df_base = df_base.collect()

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
    # streaming engine causes issues
    n_rows = df_result.select(pl.len()).collect().item()
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
    parser.add_argument('--train', type=str, help='input file', default=None)
    parser.add_argument('--test', type=str, help='input file', default=None)
    parser.add_argument('--death_record', type=str, help='input file')
    parser.add_argument('--output_path', type=str, help='input file')
    parser.add_argument('--features', action='store_true')
    args = parser.parse_args()

    # configure polars
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(10)
    rng = np.random.default_rng(123)

    # have to use rbdm, since cod_urf is only available through the end of 2020
    death = pl.scan_parquet(Path(args.death_record))
    death = death.select(['ppn_int', 'DEATH_DATE'])

    if args.test is None:
        test = False
        END_FOLLOWUP = pl.date(2018, 12, 31)
        input_file = Path(args.train)
    else:
        test = True
        END_FOLLOWUP = pl.date(2020, 1, 30)
        input_file = Path(args.test)

    df_train = pl.scan_parquet(input_file)

    # drop expensive columns to reduce memory usage during computations
    #df = df.drop(['procedure_list', 'ppn'])
    df_train = df_train.drop(['ppn'])

    df_train = preprocess_dates(df_train)

    # select only hospital episodes, the discharge episode of hospital stays
    # apdc episodes only
    apdc = df_train.filter(pl.col('apdc')==1)
    # select episodes whose end date is equal to finsep date
    apdc = df_train.filter(pl.col('episode_end_date')==pl.col('finsep'))
    # join apdc and death
    apdc = apdc.join(death, on='ppn_int', how='left', coalesce=True)
    apdc = apdc.with_columns(pl.col('DEATH_DATE').is_not_null().alias('died'))

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

    index_date_episodes = apdc.filter(pl.col('row_index')==pl.col('index_row'))
    n_unique = index_date_episodes.select(pl.col('ppn_int').n_unique()).collect().item()
    n_rows = index_date_episodes.select(pl.len()).collect().item()
    # adding index date+time as a column to be used for finding index point in esgpt data
    if n_unique != n_rows:
        print(f'Multiple index rows for one person: ppn {n_unique} and rows {n_rows}')
        raise ValueError("Rewrite program to multiple index dates returned for one patient")

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
    df_train = df_train.join(index_date, on='ppn_int', coalesce=True)

    # calculates number of days between index date and last available follow-up
    df_train = df_train.with_columns(
        (END_FOLLOWUP - pl.col('index_date'))
        .dt.total_days()
        .over('ppn_int')
        .alias('end_followup_diff')
    )
    print(df_train.select((pl.col('end_followup_diff')<0).sum()).collect())

    # sets death date as end-of-followup for those who died
    df_train = df_train.with_columns(
        pl.when(pl.col('died'))
        .then((pl.col('DEATH_DATE') - pl.col('index_date')).dt.total_days())
        .otherwise(pl.col('end_followup_diff'))
        .alias('end_followup_diff')
    )

    print(df_train.select((pl.col('end_followup_diff')<0).sum()).collect())

    # set three flags to mark inclusion in the 6months/1 year/2years sets based 
    # on available follow-up
    df_train = df_train.with_columns(
        # if more than 6 months of follow-up, or if died in the six months after index date
        ((pl.col('end_followup_diff') > 180) | pl.col('died')).alias('six_months'),
        ((pl.col('end_followup_diff') > 365) | pl.col('died')).alias('one_year'),
    )

    # remove all people with 0 or negative follow-up
    # todo: debug this, it should not be possible
    #df = df.filter(pl.col('end_followup_diff') > 0)

    #---------------------------------------------
    result_path = Path(args.output_path)
    windows = {
        'six_months': '6mo', 
        #'one_year': '1y', 
    }
    days_max = {
        '6mo': 180,
        '1y': 365,
    }

    df_train = df_train.sort(["ppn_int", "episode_start_date"])
    # make sure everyone has the six months years of follow-up
    follow_up_window = 'six_months'
    df_window = df_train.filter(pl.col(follow_up_window))

    train_test_label = 'test' if test else 'train'

    lookback_point = pl.col('index_date').dt.offset_by('-2y')
    baseline_features = get_baseline_features(
        df_window, 
        lookback_point,
        metadata_path='metadata/',
        is_train=not test,
    )
    logger.info(f'Baseline features and label being saved: {baseline_features.shape}')
    baseline_features.write_parquet(
        result_path / f'temporal_baseline_features_{follow_up_window}_{train_test_label}.parquet'
    )

    for window, offset in windows.items():
        endpoint = pl.col('index_date').dt.offset_by(offset)

        # cost
        logger.info(f'Calculating cost for {window}')
        df_cost = get_cost(df_window, endpoint=endpoint)
        (
            df_cost
            .collect(streaming=streaming_flag)
            .write_parquet(result_path / f'temporal_cost_{window}_{train_test_label}.parquet')
        )
        logger.info(df_cost.select(pl.len()).collect(streaming=streaming_flag))

        # length of stay
        logger.info(f'Calculating and saving total length of stay for {window}')
        df_los = get_length_of_stay(df_window, endpoint=endpoint, maximum=days_max[offset])
        df_los.collect(streaming=streaming_flag).write_parquet(result_path / f'temporal_length_of_stay_{window}_{train_test_label}.parquet')
        logger.info(df_los.select(pl.len()).collect(streaming=streaming_flag))

        # episode count
        logger.info(f'Calculating and saving episode count for {window}')
        df_ep_count = get_episode_count(df_window, endpoint=endpoint)
        df_ep_count.collect(streaming=streaming_flag).write_parquet(result_path / f'temporal_episode_count_{window}_{train_test_label}.parquet')
        logger.info(df_ep_count.select(pl.len()).collect(streaming=streaming_flag))

        
if __name__ == "__main__":
    main()
