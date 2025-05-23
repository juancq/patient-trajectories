import sys
import os
import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import yaml
from box import Box
from loguru import logger

import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from lightgbm import LGBMClassifier
from numpy.typing import ArrayLike
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibrationDisplay


def eval_cv_model(
    models: List[Tuple[str, Type[RegressorMixin|ClassifierMixin], Dict[str, Any]]],
    X: ArrayLike,
    y: ArrayLike,
    split: BaseCrossValidator,
    feature_set: str,
    label: str = 'label',
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
    start = time.time()
    result = []
    per_model_predictions = {}
    # for the calibration plot
    fig, ax = plt.subplots(figsize=(12,6))
    for model_name, model_cls,kwargs in models:
        scores = defaultdict(list)
        logger.info(model_name)
        print(model_name)
        predictions = []
        labels = []
        for i, (train_index,test_index) in enumerate(split.split(X, y)):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #model = model_cls()
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model_cls()),
            ])
            # use this to reduce validation overhead
            with sklearn.config_context(assume_finite=True):
                model.fit(x_train, y_train, **kwargs)
                y_pred = model.predict_proba(x_test)
                y_pred = y_pred[:, 1]

            auc = roc_auc_score(y_test, y_pred)
            ap = average_precision_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred>0.5)
            scores['auc'].append(auc)
            scores['ap'].append(ap)
            scores['accuracy'].append(accuracy)

            predictions.append(y_pred)
            labels.append(y_test)
        #logger.info(f'Mean squared error: {mse:.3f}')
        #logger.info(f'Mean absolute error: {mae:.3f}')
        print(f"AUC: {np.mean(scores['auc']):.3f} ({np.std(scores['auc']):.3f})")
        print(f"AP: {np.mean(scores['ap']):.3f} ({np.std(scores['ap']):.3f})")
        result.append({
            'task':label,
            'feature_set':feature_set,
            'model':model_name,
            'auc': np.mean(scores['auc']),
            'auc std': np.std(scores['auc']),
            'ap': np.mean(scores['ap']),
            'ap std': np.std(scores['ap']),
            'accuracy': np.mean(scores['accuracy']),
            'accuracy std': np.std(scores['accuracy']),
        })

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        per_model_predictions[f'{model_name}_y_pred'] = predictions
        per_model_predictions[f'{model_name}_y_true'] = labels
        _ = CalibrationDisplay.from_predictions(labels, predictions, 
                    name=model_name, ax=ax)

    end = time.time()
    print('Execution time', end-start, 'seconds')

    return {'metrics':result, 'predictions': per_model_predictions}


def eval_model_train_test(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    y_pred = y_pred[:, 1]

    auc = roc_auc_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred>0.5)

    logger.info(f'AUC: {auc:.3f}')
    logger.info(f'AP: {ap:.3f}')
            

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
    df_embed = df_embed.sort('ppn_int')
    n_unique = df_embed.select(pl.col('ppn_int').n_unique()).collect().item()
    len_ = df_embed.select(pl.len()).collect().item()
    logger.info(f'Embeddings unique ids and len: {n_unique}, {len_}')
    return df_embed

def threshold_label(df, task, high):
    threshold = df.select(pl.col(task).quantile(.90)).item()
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


def main():
    """
    Fit a model on dataset, where features consist of embeddings and label
    is time to death. 
    """
    pl.Config.set_tbl_cols(20)
    parser = argparse.ArgumentParser()
    #parser.add_argument('file', nargs='+', help='dataset', default=None, type=str)
    parser.add_argument('--cv', help='run cross-validation', default=False, action='store_true')
    parser.add_argument('--high', help='run cross-validation', default=False, action='store_true')
    args = parser.parse_args()
    args.cv = True

    with open('predict_config.yaml', 'r') as file:
        yaml_config = yaml.safe_load(file)
    config = Box(yaml_config)

    #parent_path = Path('/mnt/data_volume/apdc/')

    label_path = Path('/mnt/data_volume/apdc/study1/preprocessed/tasks')

    embed_path = {'esgpt':'/mnt/fast_drive/experiment/study1/pretrain/2024-08-14_14-11-29/embeddings/tuning_study1_{window}_embeddings_mean.pt',
                    'halo': label_path / '{window}_halo_embeddings_last.parquet',
                    #'esgpt': label_path / '{window}_halo_embeddings_avg_manual.parquet',
                    #'llm': label_path / '{window}_llm_embeddings.parquet',
                }

    default_kwargs = {}
    lr_params = {'solver':'saga', 'max_iter':100, 'tol': 1e-3}
    models = (
        ('LR', lambda: LogisticRegression(**lr_params),
            default_kwargs),
        ('ElasticNet', lambda: LogisticRegression(penalty='elasticnet', l1_ratio=0.5, **lr_params),
            default_kwargs),
        ('lgbm', lambda: LGBMClassifier(max_depth=8),
            default_kwargs),
        #('rf', lambda: RandomForestRegressor(n_estimators=20, max_depth=5), default_kwargs),
        ('Dummy', DummyClassifier, 
            default_kwargs),
        #('Auto', AutoML, {'model__task':'regression', 'model__time_budget':10}),
    )

    for window in config.windows:
        logger.info(f'Window: {window}')
    #for window in ['one_year']:#, 'two_year']:
        baseline_features = pl.scan_parquet(label_path / f'baseline_features_{window}.parquet')
        results_df_all = []
        for task in config.tasks:

            #embed_path = f'/mnt/fast_drive/experiment/study1/pretrain/2024-08-14_14-11-29/embeddings/tuning_study1_{window}_embeddings_mean.pt'
            #args.file = [embed_path]

            label_file = label_path / f'{task}_{window}.parquet'
            #label_file = label_path / f'{task}_{window}.parquet'
            df_label = pl.scan_parquet(label_file)
            df_label = df_label.collect(streaming=True).sample(n=config.test_size, seed=config.seed)
            label_columns = df_label.columns

            logger.info(f'Task: {task}')
            # for storing all metrics for this window, gets saved as csv
            results = []
            #for fname in sorted(args.file):
            embedding_set = {}
            for embed_name, path_to_embedding in embed_path.items():
                fname = Path(str(path_to_embedding).format(window=window))
                embedding = get_embeddings(fname)
                # taking the embeddings, and finding matches from label
                embedding = embedding.join(df_label.lazy(), on='ppn_int', coalesce=True)
                logger.info(f'Collecting {embed_name}')
                embedding = embedding.collect(streaming=True)
                embedding_set[embed_name] = embedding

            # add label to baseline features
            df_baseline_window = baseline_features.join(df_label.lazy(), on='ppn_int', coalesce=True)#, how='left')
            df_baseline_window = df_baseline_window.fill_nan(0)
            df_baseline_window = df_baseline_window.fill_null(0)
            df_baseline_window = df_baseline_window.collect(streaming=True)

            logger.info('Aligning train-test ids')
            # align any differences in set sizes in embedding set vs baseline features
            train_test_ids = df_baseline_window.select(pl.col('ppn_int'))
            for embed_name, embedding in embedding_set.items():
                train_test_ids = embedding.select(pl.col('ppn_int')).join(train_test_ids, on='ppn_int')

            logger.info('Reducing set size of models')
            for embedding_name,embedding in embedding_set.items():
                embedding = embedding.join(train_test_ids, on='ppn_int')
                embedding_set[embedding_name] = embedding.sort('ppn_int')
                print(f'{embedding_name} ', embedding.shape)
                print(embedding)
                print(embedding.select(pl.col('ppn_int').n_unique()))

            df_baseline_window = df_baseline_window.join(train_test_ids, on='ppn_int')

            df_baseline_window = threshold_label(df_baseline_window, task, args.high)

            (df_baseline_window.select(
                pl.col(task).value_counts()
                ).unnest(task)
                .write_csv(f'./plots/{window}_{task}_stats_{config.run_label}.csv')
            )
            #return

            #from IPython import embed
            #embed()
            #import code
            #code.interact(local=locals())
            #import pdb
            #pdb.set_trace()

            df_baseline_window = df_baseline_window.sort('ppn_int')
            #print('baseline', df_baseline_window.shape)

            if args.cv:
                x_base = df_baseline_window.drop(label_columns).to_numpy()
                #print(df_baseline_window.columns)
                #print(df_baseline_window.head())
                #return
                print('Baseline features', x_base.shape)

                #sys.exit()
                #continue
                if config.stratified:
                    kf = StratifiedKFold(n_splits=config.n_splits, shuffle=True,
                            random_state=config.seed)
                else:
                    kf = KFold(n_splits=config.n_splits, shuffle=True, 
                            random_state=config.seed)

                #----------------------#
                model_result = []
                for embedding_name, embedding in embedding_set.items():
                    x = embedding.select(pl.col('^feature.*$')).to_numpy()
                    print('Embeddings features', x.shape)
                    logger.info(f'Embeddings {embedding_name}')
                    y = embedding.get_column(task).to_numpy().ravel()

                    embed_model_result = eval_cv_model(models, x, y, kf, 
                                feature_set=embedding_name, label=task)
                    model_result.extend(embed_model_result['metrics'])

                    pred_name = f'plots/{task}_{window}_{embedding_name}_predictions_{config.run_label}.csv'
                    df_pred = pd.DataFrame(embed_model_result['predictions'])
                    df_pred.to_csv(pred_name)

                #----------------------#
                logger.info('Baseline')
                y = df_baseline_window.get_column(task).to_numpy().ravel()

                baseline_result = eval_cv_model(models, x_base, y, kf, 
                            feature_set='baseline', label=task)
                model_result.extend(baseline_result['metrics'])
                pred_name = f'plots/{task}_{window}_baseline_predictions_{config.run_label}.csv'
                df_pred = pd.DataFrame(baseline_result['predictions'])
                df_pred.to_csv(pred_name)

                plt.grid(True)
                # saves calibration plot
                plot_name = f'plots/{task}_{window}_{embedding_name}_{config.run_label}.png'
                plt.savefig(plot_name, bbox_inches='tight')

                results.extend(model_result)

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

            results_df = pd.DataFrame(results)
            formatted_df = results_df.copy()
            num_cols = results_df.select_dtypes(include='number').columns
            formatted_df[num_cols] = formatted_df[num_cols].map(lambda x: f'{x:.3f}')
            metric_cols = ['auc', 'ap', 'accuracy']
            for col in metric_cols:
                value = results_df[col].max()
                formatted_df[col] = np.where(results_df[col]==value,
                                formatted_df[col] + '*',
                                formatted_df[col])

            #print(formatted_df)
            results_df_all.append(formatted_df)
        results_df_all = pd.concat(results_df_all)
        high = 'high' if args.high else ''
        fout = config.result_fout_format.format(window=window, high=high)
        results_df_all.to_csv(fout)


if __name__ == "__main__":
    main()
