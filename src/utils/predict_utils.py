import yaml
from box import Box
import numpy as np
import pandas as pd
import polars as pl
from numpy.typing import ArrayLike
from typing import Dict
from loguru import logger


def threshold_label(df, task, high):
    threshold = df.select(pl.col(task).quantile(.90)).item()
    df = df.with_columns(pl.col(task).name.suffix('_raw'))
    if task == 'cost':
        df = df.with_columns(
            (pl.col(task) >= threshold).alias(task)
        )
    elif high:
        df = df.with_columns(
            (pl.col(task) >= threshold).alias(task)
        )
    else:
        df = df.with_columns(
            (pl.col(task) > 0).alias(task)
        )
    return df


def load_config(config_file='predict_config.yaml') -> Box:
    with open(config_file, 'r') as file:
        yaml_config = yaml.safe_load(file)
    return Box(yaml_config)

def build_subsets(baseline_features: pl.DataFrame, label: str=None) -> Dict[str,ArrayLike]:
    subsets = {}
    # the baseline features should include other variables, labeled as not starting with
    # "feature" in order to capture the subsets here
    # subsets are in complements
    subsets['over_75'] = baseline_features.select(pl.col('feature_age_recode') >= 75).to_numpy().flatten()
    subsets['under_75'] = baseline_features.select(pl.col('feature_age_recode') <  75).to_numpy().flatten()

    # male vs female
    subsets['female'] = baseline_features.select(pl.col('feature_sex_2').cast(pl.Boolean)).to_numpy().flatten()
    subsets['male'] = baseline_features.select(pl.col('feature_sex_1').cast(pl.Boolean)).to_numpy().flatten()

    subsets['num_episodes=1'] = baseline_features.select(pl.col('feature_num_episodes') == 1).to_numpy().flatten()
    subsets['num_episodes=2-4'] = baseline_features.select(pl.col('feature_num_episodes').is_between(2,4)).to_numpy().flatten()
    subsets['num_episodes>=5'] = baseline_features.select(pl.col('feature_num_episodes') >=5).to_numpy().flatten()

    diagnoses = {
        'cvd': 'I[1-7]',
        'diabetes': 'E1[014]',
        'ckd': 'N1[89]|E[1-4].2|I1[23]|I15.[01]|N1[124-6]|N2[5-8]|N39[12]|D59.3|B52.0|E85.3',
        #'mental_health': 'F[2-4]|X[67]|X8[0-4]',
        'mental_health': 'F[0-9]|X[67]|X8[0-4]|Y87.0',
    }

    for diagnosis_label, diagnosis_code in diagnoses.items():
        subsets[diagnosis_label] = baseline_features.select(
                (pl.col('diagnosis_history').list.eval(pl.element().str.contains(diagnosis_code)))
                .list.any()
        ).to_numpy().flatten()

    # used for the commented code
    if label:
        label = label + '_raw'
    #top1_threshold = baseline_features.select(pl.col(label).quantile(.99)).item()
    #subsets['top_1'] = baseline_features.select(pl.col(label) >= top1_threshold).to_numpy().flatten()
    #top5_threshold = baseline_features.select(pl.col(label).quantile(.95)).item()
    #subsets['top_5'] = baseline_features.select(pl.col(label) >= top5_threshold).to_numpy().flatten()

    #top20_threshold = baseline_features.select(pl.col(label).quantile(.80)).item()
    #subsets['top_20'] = baseline_features.select(pl.col(label) >= top20_threshold).to_numpy().flatten()

    for subset,data in subsets.items():
        print(f'subset {subset} - size {len(data)}, positive cases {data.sum()}')
    return subsets


def save_results(results, task, high, config, temp=None):
        results = pd.concat(results)
        # save results
        high = 'high' if high else ''
        fout = config.result_fout_format.format(task=task, high=high)
        if temp:
            fout = f'{temp}_{fout}'

        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #name, ext = fout.rsplit(".", 1)
        #fout = f"{name}_{timestamp}.{ext}"

        sort_by = ['window']
        if 'subset' in results.columns:
            sort_by.append('subset')
        results_df_all = results.sort_values(by=sort_by)
        results_df_all.to_csv(fout)


def format_results(results):
    results_df = pd.DataFrame(results)
    formatted_df = results_df.copy()
    num_cols = results_df.select_dtypes(include='number').columns
    formatted_df[num_cols] = formatted_df[num_cols].map(lambda x: f'{x:.3f}')
    metric_cols = ['auc', 'ap', 'accuracy', 'recall', 'specificity', 'brier', 'dcal']
    for col in metric_cols:
        if col in ['brier', 'dcal']:
            #value = results_df.groupby('subset')[col].min()
            value = results_df.groupby('subset')[col].transform('min')
        else:
            #value = results_df[col].max()
            #value = results_df.groupby('subset')[col].max()
            value = results_df.groupby('subset')[col].transform('max')
        formatted_df[col] = np.where(results_df[col]==value,
                            formatted_df[col] + '*',
                            formatted_df[col])
    return formatted_df


def verify_data_alignment(embedding_set: Dict[str, pl.DataFrame], df_label: pl.DataFrame) -> None:
    """
    Verify that all feature sets and labels are properly aligned by ppn_int,
    including strict row-by-row alignment checks.
    
    Parameters
    ----------
    embedding_set : Dict[str, pl.DataFrame]
        Dictionary of feature sets/embeddings
    df_label : pl.DataFrame
        DataFrame containing labels
    """
    # Get reference IDs from labels
    label_ids = df_label['ppn_int'].to_list()
    label_size = len(label_ids)
    
    # Store sizes and IDs for reporting
    sizes = {'labels': label_size}
    
    # First verify basic size and sorting
    logger.info("Performing initial size and sorting checks...")
    for name, df in embedding_set.items():
        current_size = len(df)
        sizes[name] = current_size
        
        # Verify sizes match
        if current_size != label_size:
            raise AssertionError(
                f"Size mismatch: {name} has {current_size} rows but labels have {label_size} rows"
            )
        
        # Verify sorting
        if not df['ppn_int'].is_sorted():
            raise AssertionError(f"Data not sorted by ppn_int in {name}")
    
    # Perform row-by-row alignment check
    logger.info("Performing row-by-row alignment check...")
    for i, label_id in enumerate(label_ids):
        if i % 10000 == 0 and i > 0:
            logger.info(f"Checked {i:,} rows...")
            
        for name, df in embedding_set.items():
            current_id = df['ppn_int'][i]
            if current_id != label_id:
                raise AssertionError(
                    f"Row-level misalignment detected in {name} at index {i}:\n"
                    f"Expected ppn_int {label_id}, found {current_id}"
                )
    
    # Additional check: verify exact equality of ppn_int columns
    logger.info("Performing column-level equality checks...")
    reference_ids = embedding_set[next(iter(embedding_set))]['ppn_int']
    for name, df in embedding_set.items():
        if not (df['ppn_int'] == reference_ids).all():
            raise AssertionError(
                f"Column-level misalignment detected in {name}:\n"
                "ppn_int columns are not exactly equal"
            )
    
    # Print verification summary
    logger.info("\nData alignment verification results:")
    for name, size in sizes.items():
        logger.info(f"  {name}: {size:,} rows")
    logger.info("✓ All feature sets and labels are properly aligned")
    logger.info("✓ Row-by-row alignment verified")
    logger.info("✓ Column-level equality verified")
