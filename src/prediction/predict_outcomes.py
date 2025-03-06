import sys
import argparse
from typing import Any, Dict, List, Tuple, Type
from numpy.typing import ArrayLike
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import RegressorMixin, ClassifierMixin
import sklearn
import pandas as pd
import polars as pl
import numpy as np
from collections import defaultdict
from pathlib import Path

from loguru import logger
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from flaml import AutoML

from sklearn.preprocessing import SplineTransformer


def plot_analysis(decile_stats, fname):
    plt.figure(figsize=(12,6))
    plt.plot(decile_stats['decile'], decile_stats['true_mean'], label='True Mean', marker='o')
    plt.plot(decile_stats['decile'], decile_stats['pred_mean'], label='Predicted Mean', marker='o')
    plt.xlabel('Decile')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(fname, bbox_inches='tight')


def decile_analysis(y_true, y_pred):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['decile'] = pd.qcut(df['y_true'], q=10, labels=False, duplicates='drop')
    decile_stats = df.groupby('decile').agg({
        'y_true': ['mean', 'count'],
        'y_pred': 'mean',
    }).reset_index()

    decile_stats.columns = ['decile', 'true_mean', 'count', 'pred_mean']
    decile_stats['error'] = decile_stats['pred_mean'] - decile_stats['true_mean']
    decile_stats['abs_error'] = abs(decile_stats['error'])
    decile_stats['rel_error'] = decile_stats['error'] / decile_stats['true_mean']
    return decile_stats



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
    result = []
    for model_name, model_cls,kwargs in models:
        scores = defaultdict(list)
        logger.info(model_name)
        print(model_name)
        predictions = []
        labels = []
        for i, (train_index,test_index) in enumerate(split.split(X)):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #model = model_cls()
            model = Pipeline([
                ('scaler', StandardScaler()),
                #('spline', SplineTransformer(degree=2, n_knots=3, include_bias=False)),
                ('model', model_cls()),
            ])
            # use this to reduce validation overhead
            with sklearn.config_context(assume_finite=True):
                model.fit(x_train, y_train, **kwargs)
                y_pred = model.predict(x_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            scores['mae'].append(mae)
            scores['mse'].append(mse)
            scores['r2'].append(r2)

            if i == 0 and model_name == 'ElasticNet':
                continue
                residuals = y_test - y_pred
                # plot residuals vs predicted values
                plt.figure(figsize=(10,6))
                plt.scatter(y_pred, residuals, color='blue', marker='o')
                plt.xlabel('Predicted')
                plt.ylabel('Residuals')
                plt.grid(True)
                plt.title("Residuals vs Predicted Values")
                plt.savefig(f'{label}_residuals.png', bbox_inches='tight')
                # plot predicted values vs actual values
                plt.figure(figsize=(10,6))
                plt.scatter(y_test, y_pred, color='blue', marker='o')
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.grid(True)
                plt.title("Predicted vs Actual Values")
                plt.savefig(f'{label}_pred_vs_actual.png', bbox_inches='tight')

            predictions.append(y_pred)
            labels.append(y_test)
        #logger.info(f'Mean squared error: {mse:.3f}')
        #logger.info(f'Mean absolute error: {mae:.3f}')
        print(f"Mean squared error: {np.mean(scores['mse']):.3f} ({np.std(scores['mse']):.3f})")
        print(f"Mean absolute error: {np.mean(scores['mae']):.3f} ({np.std(scores['mae']):.3f})")
        result.append({
            'task':label,
            'feature_set':feature_set,
            'model':model_name,
            'mse': np.mean(scores['mse']),
            'mse std': np.std(scores['mse']),
            'mae': np.mean(scores['mae']),
            'mae std': np.std(scores['mae']),
            'r2': np.mean(scores['r2']),
            'r2 std': np.std(scores['r2']),
        })

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        decile_stats = decile_analysis(y_true=labels, y_pred=predictions)
    return {'metrics':result, 'decile_stats':decile_stats}


def eval_model_train_test(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f'Mean squared error: {mse:.3f}')
    logger.info(f'Mean absolute error: {mae:.3f}')
            

def get_embeddings(fname: str) -> pl.DataFrame:
    """
    Reads a parquet file of embeddings

    Args:
        fname (str): The file path to the parquet file

    Returns:
        pl.DataFrame: A polars DataFrame with the embeddings. The DataFrame
        has a column named 'ppn_int' which is cast to an integer type.
    """
    logger.info(f'Reading {fname}')
    # reads embeddings
    df_embed = pl.scan_parquet(fname)
    df_embed = df_embed.with_columns(pl.col('ppn_int').cast(pl.Int32))
    df_embed = df_embed.sort('ppn_int')
    n_unique = df_embed.select(pl.col('ppn_int').n_unique()).collect().item()
    len_ = df_embed.select(pl.len()).collect().item()
    logger.info(f'Embeddings unique ids and len: {n_unique}, {len_}')
    return df_embed

def main():
    """
    Fit a model on dataset, where features consist of embeddings and label
    is time to death. 
    """
    pl.Config.set_tbl_cols(20)
    parser = argparse.ArgumentParser()
    #parser.add_argument('file', nargs='+', help='dataset', default=None, type=str)
    parser.add_argument('--cv', help='run cross-validation', default=False, action='store_true')
    parser.add_argument('--log', help='log transform run cross-validation', default=False, action='store_true')
    args = parser.parse_args()
    args.cv = True

    #parent_path = Path('/mnt/data_volume/apdc/')
    # k-fold splits
    n_splits = 5

    label_path = Path('/mnt/data_volume/apdc/study1/preprocessed/')

    embed_path = {'esgpt':'/mnt/fast_drive/experiment/study1/pretrain/2024-08-14_14-11-29/embeddings/tuning_study1_{window}_embeddings_mean.pt',
                    'halo': label_path / 'tasks' / '{window}_halo_embeddings_last.parquet',
                    #'esgpt': label_path / 'tasks' / '{window}_halo_embeddings_avg_manual.parquet',
                    #'llm': label_path / 'tasks' / '{window}_llm_embeddings.parquet',
                }

    default_kwargs = {}
    models = (
        #('LR', LinearRegression, default_kwargs), 
        ('ElasticNet', ElasticNet, default_kwargs),
        #('lgbm', LGBMRegressor, default_kwargs),
        #('rf', lambda: RandomForestRegressor(n_estimators=20, max_depth=5), default_kwargs),
        ('Dummy', lambda: DummyRegressor(strategy='median'), default_kwargs),
        #('Auto', AutoML, {'model__task':'regression', 'model__time_budget':10}),
    )

    TEST_SIZE = 50000

    for window in ['six_months', 'one_year']:#, 'two_year']:
        logger.info(f'Window: {window}')
    #for window in ['one_year']:#, 'two_year']:
        baseline_features = pl.scan_parquet(label_path / f'tasks/baseline_features_{window}.parquet')
        results_df_all = []
        for task in ['length_of_stay', 'episode_count', 'cost']:

            #embed_path = f'/mnt/fast_drive/experiment/study1/pretrain/2024-08-14_14-11-29/embeddings/tuning_study1_{window}_embeddings_mean.pt'
            #args.file = [embed_path]

            label_file = label_path / f'tasks/{task}_{window}.parquet'
            #label_file = label_path / f'{task}_{window}.parquet'
            df_label = pl.scan_parquet(label_file)
            threshold = df_label.select(pl.col(task).quantile(.90)).collect(streaming=True).item()
            df_label = df_label.with_columns(
                pl.when(pl.col(task) > threshold)
                .then(threshold)
                .otherwise(pl.col(task))
                .alias(task)
            )
            df_label = df_label.collect(streaming=True).sample(n=TEST_SIZE, seed=123)
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

            df_baseline_window = df_baseline_window.join(train_test_ids, on='ppn_int')

            #from IPython import embed
            #embed()
            #import code
            #code.interact(local=locals())
            #import pdb
            #pdb.set_trace()

            df_baseline_window = df_baseline_window.sort('ppn_int')
            print('baseline', df_baseline_window.shape)

            if args.cv:
                x_base = df_baseline_window.drop(label_columns).to_numpy()
                #print(df_baseline_window.columns)
                #print(df_baseline_window.head())
                #return
                print('Baseline features', x_base.shape)

                #sys.exit()
                #continue
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

                #----------------------#
                embed_results = []
                for embedding_name, embedding in embedding_set.items():
                    x = embedding.select(pl.col('^feature.*$')).to_numpy()
                    print('Embeddings features', x.shape)
                    logger.info(f'Embeddings {embedding_name}')
                    y = embedding.get_column(task).to_numpy().ravel()
                    # log transform label
                    if args.log:
                        y = np.log(y+0.0001)
                    embed_model_result = eval_cv_model(models, x, y, kf, 
                                feature_set=embedding_name, label=task)
                    embed_results.extend(embed_model_result['metrics'])

                #----------------------#
                logger.info('Baseline')
                y = df_baseline_window.get_column(task).to_numpy().ravel()
                # log transform label
                if args.log:
                    y = np.log(y+0.0001)
                baseline_result = eval_cv_model(models, x_base, y, kf, 
                            feature_set='baseline', label=task)

                results.extend(embed_results + baseline_result['metrics'])

                #results.extend(halo_result['metrics'] + esgpt_result['metrics'] + baseline_result['metrics'])
                #embedding_result['decile_stats'].to_csv(f'{task}_{window}_emb_decile_stats.csv')
                #baseline_result['decile_stats'].to_csv(f'{task}_{window}_baseline_decile_stats.csv')
                #return
                #plot_analysis(embedding_result['decile_stats'], f'{task}_{window}_emb_decile_stats.png')
                #plot_analysis(baseline_result['decile_stats'], f'{task}_{window}_baseline_decile_stats.png')

            else:
                raise ValueError("Implement single train-test split")
                x_train, x_test, y_train, y_test = train_test_split(df.select(pl.col('^feature.*$')), df.select(task),
                                test_size=0.20, random_state=123)

                model = LinearRegression()
                logger.info('Linear regression')
                eval_model_train_test(model, x_train, y_train, x_test, y_test)

                model = DummyRegressor()
                logger.info('Dummy regression')
                eval_model_train_test(model, x_train, y_train, x_test, y_test)
                raise ValueError("Implement writing to csv the results")

            results_df = pd.DataFrame(results)
            formatted_df = results_df.copy()
            #results_df.to_csv(f'{task}_{window}_results.csv')
            num_cols = results_df.select_dtypes(include='number').columns
            formatted_df[num_cols] = formatted_df[num_cols].map(lambda x: f'{x:.3f}')
            metric_cols = {'mse':'min', 'mae':'min', 'r2':'max'}
            for col,min_max in metric_cols.items():
                if min_max == 'min':
                    value = results_df[col].min()
                else:
                    value = results_df[col].max()
                formatted_df[col] = np.where(results_df[col]==value,
                                formatted_df[col] + '*',
                                formatted_df[col])

            #print(formatted_df)
            results_df_all.append(formatted_df)
        results_df_all = pd.concat(results_df_all)
        results_df_all.to_csv(f'{window}_results.csv')


if __name__ == "__main__":
    main()