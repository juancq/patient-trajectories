import numpy as np
from sklearnex import config_context as sklearnex_config_context
from sklearnex import patch_sklearn

patch_sklearn()

import argparse
import os
import pickle
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

import utils.predict_utils as ut
from utils.kill import conditional_set_memory_limit
from evaluation.metrics import calculate_metrics

streaming_flag = False


def evaluate_subsets(subset_data, y_test, y_pred, test_index, scores):
    for subset_name, subset_mask in subset_data.items():
        subset_y_pred = y_pred[subset_mask[test_index]]
        subset_y_test = y_test[subset_mask[test_index]]

        # calculate metrics for this fold
        metrics = calculate_metrics(y_test=subset_y_test, y_pred=subset_y_pred, threshold=0.5)
        # Store metrics and predictions
        for metric_name, metric_value in metrics.items():
            scores[subset_name][metric_name].append(metric_value)


def eval_cv_model(
    models: List[Tuple[str, Type[RegressorMixin|ClassifierMixin], Dict[str, Any]]],
    X: ArrayLike,
    y: ArrayLike,
    fold_indices: List[Tuple[ArrayLike,ArrayLike]],
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
    X : ArrayLike
        The feature array.
    y : ArrayLike
        The target array.
    split : BaseCrossValidator
        The cross-validation object.
    feature_set : str
        The name of the feature set.
    label : str, optional
        The label for the task, by default 'label'.
    """
    cv_start = time.time()
    result = []
    per_model_predictions = {}
    # for the calibration plot
    #fig, ax = plt.subplots(figsize=(12,6))
    save_dir = Path(f'folds/{label}/{window}/')
    save_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model_cls,kwargs in models:
        scores = defaultdict(lambda: defaultdict(list))
        logger.info(model_name)
        print(model_name)
        predictions = []
        labels = []
        for i, (train_index,test_index) in enumerate(tqdm(fold_indices)):
            fold_start = time.time()
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #model = model_cls()
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model_cls()),
            ])
            # use this to reduce validation overhead
            with sklearnex_config_context():#target_offload="gpu:0"):
                with sklearn.config_context(assume_finite=True):
                    model.fit(x_train, y_train, **kwargs)
                    save_data = {
                        'model': model,
                        'metadata': {
                            'feature_set': feature_set,
                            'path': str(save_dir),
                            'task': label,
                            'fold': i,
                        }
                    }
                    fold_dir = save_dir / f'fold_{i}'
                    fold_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(save_data, fold_dir / f'{feature_set}_{model_name}.joblib')
                    y_pred = model.predict_proba(x_test)
                    y_pred = y_pred[:, 1]

                    # save predictions to fold folder
                    df_pred = pd.DataFrame({'prediction':y_pred, 'true':y_test})
                    df_pred.to_csv(fold_dir / f'{feature_set}_{model_name}_predictions.csv')

            # calculate metrics for this fold
            metrics = calculate_metrics(y_test, y_pred, threshold=0.5)
            # Store metrics and predictions
            for metric_name, metric_value in metrics.items():
                scores['full_test'][metric_name].append(metric_value)

            evaluate_subsets(
                subset_data, y_test=y_test, y_pred=y_pred, 
                test_index=test_index, scores=scores
            )

            fold_end = time.time()
            fold_runtime = fold_end - fold_start
            print(f'\nFold Execution time {fold_runtime:.1f} seconds or {fold_runtime/60:.1f} minutes')

        #logger.info(f'Mean squared error: {mse:.3f}')
        #logger.info(f'Mean absolute error: {mae:.3f}')
        print(f"AUC: {np.mean(scores['full_test']['auc']):.3f} ({np.std(scores['full_test']['auc']):.3f})")
        print(f"AP: {np.mean(scores['full_test']['ap']):.3f} ({np.std(scores['full_test']['ap']):.3f})")


        for subset_name, subset_scores in scores.items():
            # Compile results
            fold_results = {
                'subset': subset_name,
                'task':label,
                'feature_set':feature_set,
                'model':model_name,
            }

            # add mean and std for each metric
            for metric_name, metric_values in subset_scores.items():
                if metric_name == 'dcal_pvalue':
                    fold_results[metric_name] = np.sum(metric_values)
                    fold_results[f'{metric_name} std'] = 0
                else:
                    fold_results[metric_name] = np.mean(metric_values)
                    fold_results[f'{metric_name} std'] = np.std(metric_values)
                fold_results[f'{metric_name}_range'] = metric_values
        
            # add result for full-test and each subset
            result.append(fold_results)

        #_ = CalibrationDisplay.from_predictions(labels, predictions, 
        #            name=model_name, ax=ax)

    cv_end = time.time()
    cv_runtime = cv_end - cv_start
    print(f'CV Execution time {cv_runtime:.1f} seconds or {cv_runtime/60:.1f} minutes')

    return {'metrics':result}


def eval_model_train_test(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred = y_pred[:, 1]

    metrics = calculate_metrics(y_test, y_pred, threshold=0.5)

    logger.info(f"AUC: {metrics['auc']:.3f}")
    logger.info(f"AP: {metrics['ap']:.3f}")
            

def add_diagnosis_codes(df, filter_codes=None):
    # convert all procedure codes to counts/dummy variables
    print(df.height)
    df_codes = df.select(pl.col(['ppn_int', 'diagnosis_history'])).lazy()
    #icd_codes = df_codes.select('diagnosis_history').explode('diagnosis_history').unique().collect().get_column('diagnosis_history').to_list()
    #print(len(icd_codes))
    #sys.exit()
    df_codes = df_codes.explode('diagnosis_history')
    print(df_codes)
    code_counts = df_codes.select(pl.col('diagnosis_history').value_counts()).collect().unnest('diagnosis_history')
    if filter_codes is not None:
        code_counts = code_counts.filter(pl.col('count') > filter_codes)
        code_counts = code_counts.filter(pl.col('diagnosis_history').is_not_null())
        df_codes = df_codes.filter(pl.col('diagnosis_history').is_in(code_counts.get_column('diagnosis_history').unique()))
    #print(len(code_counts))
    #print(len(code_counts))
    #import code
    #code.interact(local=locals())

    #icd_codes = df_codes.select('diagnosis_history').explode('diagnosis_history').unique().collect().get_column('diagnosis_history').to_list()
    #icd_codes.remove(None)
    ##print(icd_codes)
    #agg_expr = [
    #    pl.col('diagnosis_history').list.contains(h).cast(pl.Int64).sum().alias(h) for h in icd_codes
    #]
    #df_codes = df_codes.group_by(['ppn_int']).agg(agg_expr)
    #df_codes = df_codes.collect(streaming=streaming_flag)

    #print(df_codes.head(20))
    #sys.exit()

    #df_codes = df_codes.with_columns(pl.col('diagnosis_history').str.split('.').list.get(0).alias('diagnosis_history'))
    #df_codes = df_codes.with_columns(pl.col('diagnosis_history').str.extract(r"([^.]+\.\d|^[^.]+)").alias('diagnosis_history'))
    # to get diagnosis counts in lookback history
    df_codes = df_codes.group_by(['ppn_int', 'diagnosis_history']).agg(
        value=pl.len()
    )#.collect(streaming=streaming_flag)
    #print(df_codes.select(pl.len()).collect())
    #print(df_codes.head(20).collect())

    #print(df_codes.head(20).collect())
    #print(df_codes.select(pl.len()).collect())
    #sys.exit()
    df_codes = df_codes.collect(streaming=streaming_flag)

    # diagnosis dummy variable (yes/no in lookback history)
   # df_codes = df_codes.with_columns(value=pl.lit(1)).collect(streaming=streaming_flag)

    print(df_codes.head(20))
    df_codes = df_codes.pivot(on="diagnosis_history", index='ppn_int', values='value')
    print(df_codes.head(20))
    df_codes = df_codes.select(pl.all().shrink_dtype())
    # add "diagnosis" prefix to all diagnosis columns
    df_codes = df_codes.select(pl.col('ppn_int'), cs.exclude('ppn_int').name.prefix('feature_diagnosis_'))
    print(df_codes.head(20))
    print(df_codes.height)

    # add procedure code to other baseline variables
    df = df.join(df_codes, on='ppn_int', coalesce=True, how='left')
    df = df.with_columns(cs.starts_with('feature_diagnosis').fill_null(0))
    print(df.head(20))
    print(df.height)
    return df


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
    lr_params = {'solver':'saga', 'max_iter':100, 'tol': 1e-3}
    gbt_params = {
        'learning_rate': 0.05,
        'n_estimators': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 10,
    }
    return [
        ('LR', lambda: LogisticRegression(**lr_params), default_kwargs),
        ('lgbm', lambda: LGBMClassifier(**gbt_params), default_kwargs),
        ('ElasticNet', lambda: LogisticRegression(penalty='elasticnet', l1_ratio=0.5, **lr_params), default_kwargs),
        ('Dummy', DummyClassifier, default_kwargs),
    ]

def process_embeddings(embed_path, window, label_ids):
    embedding_set = {}
    for embed_name, path_to_embedding in embed_path.items():
        fname = Path(str(path_to_embedding).format(window=window))
        embedding = get_embeddings(fname)
        embedding = label_ids.join(embedding, on='ppn_int', coalesce=True)
        embedding = embedding.sort('ppn_int')
        logger.info(f'Collecting {embed_name}')
        embedding = embedding.collect(streaming=streaming_flag)
        # stats
        n_unique = embedding.select(pl.col('ppn_int').n_unique()).item()
        len_ = embedding.select(pl.len()).item()
        logger.info(f'Embeddings after joining with label unique ids and len: {n_unique}, {len_}')

        embedding_set[embed_name] = embedding
    return embedding_set


def perform_cross_validation(embedding_set, task, fold_indices, 
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

        x = embedding.select(cs.starts_with('feature'))
        if embedding_name == 'baseline':
            # for standard baseline, do not include new variables
            x = x.select(cs.exclude(cs.starts_with('feature_new')))
        x = x.to_numpy()

        print(f'{embedding_name} features', x.shape)
        logger.info(f'{embedding_name}')
        y = embedding.get_column(task).to_numpy().ravel()

        embed_model_result = eval_cv_model(
            models_to_eval, 
            x, y, fold_indices, 
            window,
           feature_set=embedding_name, label=task,
           subset_data=subset_data
        )
        result.extend(embed_model_result['metrics'])

    #plt.grid(True)
    #plot_name = f'plots/{task}_{window}_calibration_{config.run_label}.png'
    #plt.savefig(plot_name, bbox_inches='tight')

    return result



@conditional_set_memory_limit(24)
def main():
    """
    Fit a model on dataset, where features consist of embeddings and label
    is time to death. 
    """
    main_start_time = time.time()
    pl.Config.set_tbl_cols(20)
    parser = argparse.ArgumentParser()
    #parser.add_argument('file', nargs='+', help='dataset', default=None, type=str)
    parser.add_argument('--no_cv', help='run cross-validation', default=False, action='store_true')
    parser.add_argument('--high', help='predict 90% percentile', default=True, action='store_true')
    args = parser.parse_args()

    config = ut.load_config('config/predict_config.yaml')

    w_temp = 'two_year'
    if os.name == 'nt':
        label_path = Path(config.label_path_nt)
        embed_path = {
            #'esgpt':label_path / 'held_out_study1_two_year_embeddings_mean.pt',
            #'esgpt_new':label_path / 'study1_held_out_two_year_embeddings_mean_loss_15.84.pt',
            #'llm':label_path / 'two_year_minilm_l6_embeddings.parquet',
            'llm_e5':label_path / 'two_year_multilingual-e5-large-instruct_embeddings.parquet',
            #'halo': label_path / 'two_year_halo_embeddings_last_original_epoch34.parquet',
            #'halo_n48': label_path / f'{w_temp}_halo_embeddings_last_n48_v24.parquet',
            #'halo_n48_v29': label_path / f'{w_temp}_halo_embeddings_last_v29.parquet',
            #'halo_old': label_path / 'two_year_halo_embeddings_last_old.parquet',
            #'esgpt': label_path / '{window}_halo_embeddings_avg_manual.parquet',
            #'llm': label_path / '{window}_llm_embeddings.parquet',
        }
        #embed_path = {
        #                'halo_original': label_path / f'{w_temp}_halo_embeddings_last_original.parquet',
        #                'halo_old': label_path / f'{w_temp}_halo_embeddings_last_old.parquet',
        #                #'halo_n48_epoch16': label_path / f'{w_temp}_halo_embeddings_last_n48_epoch16.parquet',
        #                #'halo_original_epoch0': label_path / f'{w_temp}_halo_embeddings_last_original_epoch0.parquet',
        #            }
    else:
        label_path = Path(config.label_path_linux)
        embed_path = {'esgpt':'/mnt/fast_drive/experiment/study1/pretrain/2024-08-14_14-11-29/embeddings/tuning_study1_{window}_embeddings_mean.pt',
                        #'halo': label_path / f'{w_temp}_halo_embeddings_last_v1.parquet',
                        #'halo_n48': label_path / f'{w_temp}_halo_embeddings_last_n48_v24.parquet',
                        'halo_n48_v29': label_path / f'{w_temp}_halo_embeddings_last_v29.parquet',
                        #'esgpt': label_path / '{window}_halo_embeddings_avg_manual.parquet',
                        #'llm': label_path / '{window}_llm_embeddings.parquet',
                   }

    models = get_models()

    baseline_features = pl.scan_parquet(label_path / f'baseline_features_strong_two_year.parquet')
    #df_baseline_window = label_ids.join(baseline_features, on='ppn_int', coalesce=True)
    baseline_features = baseline_features.fill_nan(0)
    baseline_features = baseline_features.fill_null(0)
    n_unique = baseline_features.select(pl.col('ppn_int').n_unique()).collect().item()
    len_ = baseline_features.select(pl.len()).collect().item()
    logger.info(f'Baseline features unique ids and len: {n_unique}, {len_}')

    ids = baseline_features.select(pl.col('ppn_int'))
    if config.test_size is not None: 
        test_size = int(config.test_size)
        if test_size <=0:
            raise ValueError(f'Test size must be a positive number: {test_size}')
        # for exploration, use subset to run fast
        ids = ids.collect(streaming=streaming_flag).sample(n=test_size, seed=config.seed)
        logger.info(f'Baseline features using ids {ids.select(pl.len()).item()}')
        ids = ids.lazy()

    #baseline_features = baseline_features.collect(streaming=streaming_flag)

    if config.stratified:
        kf = StratifiedKFold(n_splits=config.n_splits, shuffle=True,
                random_state=config.seed)
    else:
        kf = KFold(n_splits=config.n_splits, shuffle=True, 
                random_state=config.seed)

    # process each task
    for task in config.tasks:
        logger.info(f'Task: {task}')
        results_df_all = []

        # process each prediction horizon
        for window_index, window in enumerate(config.windows):
            logger.info(f'Window: {window}')
            print(f'Window: {window}')

            # read label file
            df_label = pl.scan_parquet(label_path / f'{task}_{window}.parquet')
            # keep only ppns present in all horizons
            df_label = ids.join(df_label, on='ppn_int', how='inner')
            df_label = df_label.sort('ppn_int')
            df_label = df_label.collect(streaming=streaming_flag)
            label_ids = df_label.lazy().select('ppn_int')

            # get the embeddings of only the ppns in label dataframe
            embedding_set = process_embeddings(embed_path, window, label_ids)

            # baseline features
            df_baseline_window = label_ids.join(baseline_features, on='ppn_int', coalesce=True)
            df_baseline_window = df_baseline_window.collect(streaming=streaming_flag)
            embedding_set['baseline'] = df_baseline_window

            # convert task to binary
            df_label = ut.threshold_label(df_label, task, args.high)
            df_label = df_label.sort('ppn_int')
            print('number of labels', df_label)

            logger.info('Reducing set size of models')
            # add label to embeddings
            print('^'*20)
            label_columns = df_label.columns

            for embedding_name,embedding in embedding_set.items():
                embedding = df_label.join(embedding, on='ppn_int')
                embedding = embedding.sort('ppn_int')
                print(f'{embedding_name} ', embedding.shape)
                print(embedding)
                print(embedding.select(pl.col('ppn_int').n_unique()))
                embedding_set[embedding_name] = embedding
                #embedding.select(label_columns).write_csv(f'{task}_{window}.csv')

            ut.verify_data_alignment(embedding_set, df_label)

            (df_label.select(
                pl.col(task).value_counts()
                ).unnest(task)
                .write_csv(f'./plots/{window}_{task}_stats_{config.run_label}.csv')
            )

            subset_data = ut.build_subsets(embedding_set['baseline'], task)

            # diagnosis codes
            embedding_set['baseline'] = add_diagnosis_codes(embedding_set['baseline'])
            embedding_set['baseline'] = embedding_set['baseline'].with_columns(
                (cs.starts_with('feature_procedure_') > 0).cast(pl.Boolean),
                (cs.starts_with('feature_diagnosis_') > 0).cast(pl.Boolean),
            )
            
            # the new baseline points to baseline data
            # new vs old baseline variable selection done in validation loop
            #embedding_set['baseline_new'] = embedding_set['baseline']

            # for storing all metrics for this window, gets saved as csv
            window_results = []
            indices_label = 'diagnosis'
            if not args.no_cv:

                # calculate indexes for cross-validation and save to file
                X = embedding_set['baseline'].select('ppn_int').with_row_index()
                y = embedding_set['baseline'].get_column(task)
                fold_indices = list(kf.split(X, y))
                save_indices = {
                    'fold_indices': fold_indices,
                    'subset_data': subset_data
                }
                save_dir = Path(f'folds/{task}/{window}/')
                save_dir.mkdir(parents=True, exist_ok=True)
                with open(save_dir / f'fold_indices_{indices_label}.pkl', 'wb') as f:
                    pickle.dump(save_indices, f)
                save_index = []
                for train_index, test_index in fold_indices:
                    train_data = X.filter(pl.col('index').is_in(train_index)).select(pl.col(['index', 'ppn_int']))
                    test_data = X.filter(pl.col('index').is_in(test_index)).select(pl.col(['index', 'ppn_int']))
                    fold = {'train': train_data, 'test': test_data}
                    save_index.append(fold)

                with open(save_dir / f'fold_indices_ppn_{indices_label}.pkl', 'wb') as f:
                    pickle.dump(save_index, f)

                embedding_set = {'baseline_diagnosis': embedding_set['baseline'].select(cs.starts_with('feature'), pl.col(task))}
                # remove this line in the future
                #del embedding_set['baseline']
                start_time = time.time()

                # clean memory
                del X
                del y

                cv_results = perform_cross_validation(
                    embedding_set, task, fold_indices, models, config, window,
                    subset_data=subset_data
                )
                end_time = time.time()
                runtime = end_time - start_time
                print(f'Execution time of task {task} window {window}: {runtime:.1f} seconds or {runtime/60:.1f} minutes')

                window_results.extend(cv_results)

            else:
                raise ValueError("Implement single train-test split")
                x_train, x_test, y_train, y_test = train_test_split(df.select(pl.col('^feature.*$')), df.select(task),
                                test_size=0.20, random_state=config.seed)

                model = LogisticRegression()
                logger.info('Logistic regression')
                eval_model_train_test(model, x_train, y_train, x_test, y_test)

                model = DummyClassifier()
                logger.info('Dummy')
                eval_model_train_test(model, x_train, y_train, x_test, y_test)
                raise ValueError("Implement writing to csv the results")

            formatted_df = ut.format_results(window_results)
            formatted_df['window'] = window
            results_df_all.append(formatted_df)
            # save results
            ut.save_results(results_df_all, task, args.high, config, temp='temp')

        # save results
        ut.save_results(results_df_all, task, args.high, config)

    main_end_time = time.time()
    runtime = main_end_time - main_start_time
    print(f'Execution time of script: {runtime/60:.1f} minutes')


if __name__ == "__main__":
    main()
