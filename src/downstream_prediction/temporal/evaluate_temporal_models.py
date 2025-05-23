import numpy as np
from sklearnex import config_context as sklearnex_config_context
from sklearnex import patch_sklearn

patch_sklearn()

import argparse
import os
import pickle
import copy
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import polars.selectors as cs
import sklearn
import yaml
from box import Box
from lightgbm import LGBMClassifier
from loguru import logger
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (BaseCrossValidator, KFold,
                                     StratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import predict_utils as ut
from metrics import calculate_metrics

streaming_flag = False
SEED = 123
np.random.seed(SEED)


def conditional_set_memory_limit(value):
    def wrapper(func):
        if os.name == 'nt':
            from kill import set_memory_limit
            return set_memory_limit(value)(func)
        return func
    return wrapper


def evaluate_subsets(subset_data, y_test, y_pred, scores):
    for subset_name, subset_mask in subset_data.items():
        subset_y_pred = y_pred[subset_mask]
        subset_y_test = y_test[subset_mask]

        # calculate metrics for this fold
        metrics = calculate_metrics(y_test=subset_y_test, y_pred=subset_y_pred, threshold=0.5)
        # Store metrics and predictions
        for metric_name, metric_value in metrics.items():
            scores[subset_name][metric_name] = metric_value

def evaluate_test_set(model, x_test, y_test, scores, subset_data):
    y_pred = model.predict_proba(x_test)
    y_pred = y_pred[:, 1]
    # calculate metrics for this fold
    metrics = calculate_metrics(y_test, y_pred, threshold=0.5)
    # Store metrics and predictions
    for metric_name, metric_value in metrics.items():
        scores['full_test'][metric_name].append(metric_value)

    evaluate_subsets(
        subset_data, y_test=y_test, y_pred=y_pred, 
        scores=scores
    )

def evaluate_test_set_no_pred(y_test, y_pred, scores, subset_data):
    # calculate metrics for this fold
    metrics = calculate_metrics(y_test, y_pred, threshold=0.5)
    # Store metrics and predictions
    for metric_name, metric_value in metrics.items():
        scores['full_test'][metric_name].append(metric_value)

    evaluate_subsets(
        subset_data, y_test=y_test, y_pred=y_pred, 
        scores=scores
    )

def bootstrap_test_set(
    y_test, y_pred, bootstrap_indexes, num_points, scores, subset_data
    ):
    logger.info(f'Performing {len(bootstrap_indexes)} bootstraps on test set')
    for bootstrap_i in tqdm(bootstrap_indexes):
        bootstrap_y_test = y_test[bootstrap_i]
        bootstrap_y_pred = y_pred[bootstrap_i]

        evaluate_test_set_no_pred(
            bootstrap_y_test, bootstrap_y_pred, scores, subset_data
        )


def bootstrap_train_set():
    n_bootstrap = 100
    bootstrap_indexes = np.random.randint(
        low=0, high=num_points, size=(n_bootstrap,num_points)
    )

    for model_name, model_cls,kwargs in models:
        scores = defaultdict(lambda: defaultdict(list))
        logger.info(model_name)
        print(model_name)

        per_model_start = time.time()

        print(f'{label} features', x_train.shape)
        logger.info(f'{label}')

        model = model_cls()

        logger.info(f'Performing {n_bootstrap} bootstraps')
        for bootstrap_idx in tqdm(bootstrap_indexes):
            # use this to reduce validation overhead
            with sklearnex_config_context():#target_offload="gpu:0"):
                with sklearn.config_context(assume_finite=True):
                    model.fit(x_train[bootstrap_idx], y_train[bootstrap_idx], **kwargs)

            evaluate_test_set(model, x_test, y_test, scores, subset_data)


def evaluate_model(
    models: List[Tuple[str, Type[RegressorMixin|ClassifierMixin], Dict[str, Any]]],
    embedding: dict,
    window: str,
    feature_set: str,
    label: str = 'label',
    subset_data: Dict[str, ArrayLike] = None,
) -> Dict[str, Any]:
    """
    Evaluate a list of models in a cross-validation setting.

    Parameters
    ----------
    models : List[Tuple[str, Type[RegressorMixin], Dict[str, Any]]]
        A list of tuples containing the model name, class, and keyword arguments.
    feature_set : str
        The name of the feature set.
    label : str, optional
        The label for the task, by default 'label'.
    """
    cv_start = time.time()
    result = []

    x_train = embedding['train'].select(cs.starts_with('feature')).to_numpy().astype(np.half)
    y_train = embedding['train'].get_column(label).to_numpy().ravel()
    del embedding['train']

    models_trained = {}

    for model_name, model_cls,kwargs in models:

        logger.info(model_name)

        per_model_start = time.time()

        logger.info(f'{label} features - {feature_set}', x_train.shape)

        model = model_cls()

        # use this to reduce validation overhead
        with sklearnex_config_context():#target_offload="gpu:0"):
            with sklearn.config_context(assume_finite=True):
                model.fit(x_train, y_train, **kwargs)
                models_trained[model_name] = model
                #save_data = {
                #    'model': model,
                #    'metadata': {
                #        'feature_set': feature_set,
                #        'path': str(save_dir),
                #        'task': label,
                #        'fold': i,
                #    }
                #}
                #joblib.dump(save_data, fold_dir / f'{feature_set}_{model_name}.joblib')

        per_model_end = time.time()
        per_model_runtime = per_model_end - per_model_start
        print(f'\nPer model train time {per_model_runtime:.1f} seconds or {per_model_runtime/60:.1f} minutes')

    del x_train
    del y_train
    x_test = embedding['test'].select(cs.starts_with('feature')).to_numpy().astype(np.half)
    y_test = embedding['test'].get_column(label).to_numpy().ravel()
    num_points = len(x_test)

    n_bootstraps = 100
    # generate bootstrap indexes
    bootstrap_indexes = np.random.randint(
        low=0, high=num_points, size=(n_bootstraps, num_points)
    )

    # evaluate the trained models; splitting train and prediction into separate
    # loops for memory management
    for model_name, model_cls,kwargs in models:
        # nested dictionary (subset: metric: [list of metric scores across bootstraps])
        scores = defaultdict(lambda: defaultdict(list))

        model = models_trained[model_name]
        y_pred = model.predict_proba(x_test)
        y_pred = y_pred[:, 1]

        # save predictions for calibration plot
        df_predictions = pl.DataFrame(
            {
                'prediction':y_pred, 
                'true':y_test,
            }
        )
        df_predictions.write_csv(f'temporal/temporal_{feature_set}_{label}_{window}_predictions.csv')

        # perform bootstrap
        bootstrap_test_set(
            y_test, y_pred, bootstrap_indexes, num_points, scores, subset_data
        )

        # Compile results
        for subset_name, subset_scores in scores.items():
            bootstrap_results = {
                'subset': subset_name,
                'task':label,
                'feature_set':feature_set,
                'model':model_name,
            }

            # add mean and std for each metric
            for metric_name, metric_values in subset_scores.items():
                if metric_name == 'dcal_pvalue':
                    bootstrap_results[metric_name] = np.sum(metric_values)
                    bootstrap_results[f'{metric_name} std'] = 0
                else:
                    bootstrap_results[metric_name] = np.mean(metric_values)
                    bootstrap_results[f'{metric_name} std'] = np.std(metric_values)
                    lower = np.percentile(metric_values, 2.5)
                    higher = np.percentile(metric_values, 97.5)
                    bootstrap_results[f'{metric_name}_CI'] = f'({lower:0.3f}-{higher:0.3f})'
                bootstrap_results[f'{metric_name}_range'] = metric_values
        
            # add result for full-test and each subset
            result.append(bootstrap_results)


    cv_end = time.time()
    cv_runtime = cv_end - cv_start
    print(f'CV Execution time {cv_runtime:.1f} seconds or {cv_runtime/60:.1f} minutes')
    #plt.close()

    return {'metrics':result}


def get_embeddings(fname: str) -> pl.DataFrame:
    """
    Reads a parquet file of embeddings

    Args:
        fname (str): The file path to the parquet file

    Returns:
        pl.DataFrame: A polars DataFrame with the embeddings. The DataFrame
        has a column named 'ppn_int' which is cast to an integer type.
    """
    if not os.path.isfile(fname):
        raise ValueError(f'Invalid embedding path: {fname}')

    logger.info(f'Reading {fname}')
    # reads embeddings
    df_embed = pl.scan_parquet(fname)
    df_embed = df_embed.with_columns(pl.col('ppn_int').cast(pl.Int32))
    n_unique = df_embed.select(pl.col('ppn_int').n_unique()).collect().item()
    len_ = df_embed.select(pl.len()).collect().item()
    logger.info(f'Embeddings unique ids and len: {n_unique}, {len_}')
    return df_embed


def get_models():
    default_kwargs = {}
    #lr_params = {'solver':'saga', 'max_iter':100, 'tol': 1e-3}
    lr_params = {'solver':'saga', 'max_iter':150, 'tol': 1e-3}
    gbt_params = {
        #'learning_rate': 0.05,
        #'n_estimators': 100,
        #'feature_fraction': 0.8,
        #'bagging_fraction': 0.8,
        #'bagging_freq': 5,
        #'max_depth': 8,
        'random_state': SEED,
        #'histogram_pool_size': 16384,
    }
    return [
        #('LR', lambda: LogisticRegression(**lr_params), default_kwargs),
        #('ElasticNet', lambda: LogisticRegression(penalty='elasticnet', l1_ratio=0.5, **lr_params), default_kwargs),
        ('lgbm', lambda: LGBMClassifier(**gbt_params), default_kwargs),
        #('Dummy', DummyClassifier, default_kwargs),
    ]


def perform_single_split_validation(
    embedding_set, task,
    models, config, window,
    subset_data: Dict[str, ArrayLike] = None
    ):
    result = []
    for embedding_name, embedding in embedding_set.items():
        models_to_eval = models

        if 'baseline' not in embedding_name:
            # only run dummy classifier for baseline, as dummy classifier does not look at input features
            models_to_eval = [m for m in models_to_eval if 'dummy' not in m[0].lower()]
            #continue

        embed_model_result = evaluate_model(
            models_to_eval, 
            embedding,
            window,
           feature_set=embedding_name, label=task,
           subset_data=subset_data
        )
        result.extend(embed_model_result['metrics'])

    return result


def get_baseline_features(filepath, is_train, config):
    filepath = Path(filepath)
    train_test = 'train' if is_train else 'test'
    baseline_features_train = pl.scan_parquet(filepath / config[train_test])
    baseline_features_train = baseline_features_train.fill_nan(0)
    baseline_features_train = baseline_features_train.fill_null(0)
    n_unique = baseline_features_train.select(pl.col('ppn_int').n_unique()).collect().item()
    len_ = baseline_features_train.select(pl.len()).collect().item()
    logger.info(f'Baseline features unique ids and len: {n_unique}, {len_}')
    return baseline_features_train


@conditional_set_memory_limit(29)
def main():
    """
    Fit a model on dataset, where features consist of embeddings and label
    is time to death. 
    """
    main_start_time = time.time()
    pl.Config.set_tbl_cols(20)
    parser = argparse.ArgumentParser()
    #parser.add_argument('file', nargs='+', help='dataset', default=None, type=str)
    #parser.add_argument('--baseline_path', type=str, help='input file')
    #parser.add_argument('--train', type=str, help='input file')
    #parser.add_argument('--test', type=str, help='input file')
    parser.add_argument('--config', type=str, help='input file')
    high_cost_binary = True
    
    args = parser.parse_args()

    config = ut.load_config(args.config)

    models = get_models()

    logger.info('Getting baseline train features')
    baseline_features_train = get_baseline_features(
        config['baseline']['path'],
        is_train=True,
        config=config['baseline'],
    )

    logger.info('Getting baseline test features')
    baseline_features_test = get_baseline_features(
        config['baseline']['path'],
        is_train=False,
        config=config['baseline'],
    )

    config['halo_embeddings']['path'] = Path(config['halo_embeddings']['path'])
    halo_embed_train = pl.scan_parquet(config['halo_embeddings']['path'] / config['halo_embeddings']['train'])
    halo_embed_test = pl.scan_parquet(config['halo_embeddings']['path'] / config['halo_embeddings']['test'])

    embedding_set_default = {
        'baseline':
        {
            'train': baseline_features_train,
            'test': baseline_features_test,
        },
        'halo':
        {
            'train': halo_embed_train,
            'test': halo_embed_test,
        }
    }


    # process each task
    for task, train_test_set in config.tasks.items():
        logger.info(f'Task: {task}')
        results_df_all = []

        # process each prediction horizon
        for window_index, window in enumerate(config.windows):
            logger.info(f'Window: {window}')
            print(f'Window: {window}')

            # read label file
            label_path = Path(config.tasks_path)
            df_label = pl.scan_parquet(label_path / train_test_set['train'])
            df_label = df_label.collect(streaming=True)

            # convert task to binary
            df_label, task_threshold = ut.threshold_label(df_label, task, high=high_cost_binary, threshold=None)
            #df_label = df_label.sort('ppn_int')
            print('number of labels', df_label)

            df_label_test = pl.scan_parquet(label_path / train_test_set['test'])
            df_label_test = df_label_test.collect(streaming=True)
            df_label_test, _ = ut.threshold_label(df_label_test, task, high=high_cost_binary, threshold=task_threshold)
            #df_label_test = df_label_test.sort('ppn_int')

            df_label = df_label.sample(
                fraction=1.0, 
                with_replacement=False, 
                shuffle=True,
                seed=SEED,
            )
            df_label_test = df_label_test.sample(
                fraction=1.0, 
                with_replacement=False, 
                shuffle=True,
                seed=SEED,
            )

            logger.info('Reducing set size of models')
            # add label to embeddings
            print('^'*20)

            embedding_set = copy.deepcopy(embedding_set_default)
            for feature_set, embed in embedding_set.items():
                embed['train'] = embed['train'].collect(streaming=True)
                embed['test'] = embed['test'].collect(streaming=True)

            # add label to embedding
            for embedding_name,embedding_train_test in embedding_set.items():
                embedding = df_label.join(embedding_train_test['train'], on='ppn_int')
                print(f'{embedding_name} ', embedding.shape)
                print(embedding)
                print(embedding.select(pl.col('ppn_int').n_unique()))
                embedding_set[embedding_name]['train'] = embedding

                embedding = df_label_test.join(embedding_train_test['test'], on='ppn_int')
                embedding_set[embedding_name]['test'] = embedding

            #ut.verify_data_alignment(
            #    {name: emb['train'] for name,emb in embedding_set.items()}, 
            #    df_label
            #)
            #ut.verify_data_alignment(
            #    {name: emb['test'] for name,emb in embedding_set.items()}, 
            #    df_label_test
            #)

            #(df_label.select(
            #    pl.col(task).value_counts()
            #    ).unnest(task)
            #    .write_csv(f'./plots/{window}_{task}_stats_{config.run_label}.csv')
            #)

            subset_data = ut.build_subsets(embedding_set['baseline']['test'], task)

            # for storing all metrics for this window, gets saved as csv
            start_time = time.time()

            window_results = perform_single_split_validation(
                embedding_set, task, models, config, window,
                subset_data=subset_data
            )
            end_time = time.time()
            runtime = end_time - start_time
            print(f'Execution time of task {task} window {window}: {runtime:.1f} seconds or {runtime/60:.1f} minutes')

            formatted_df = ut.format_results(window_results)
            formatted_df['window'] = window
            results_df_all.append(formatted_df)
            # save results
            #ut.save_results(results_df_all, task, high_cost_binary, config, temp='temp')

            del embedding_set

        # save results
        ut.save_results(results_df_all, task, high_cost_binary, config)

    main_end_time = time.time()
    runtime = main_end_time - main_start_time
    print(f'Execution time of script: {runtime/60:.1f} minutes')


if __name__ == "__main__":
    main()
