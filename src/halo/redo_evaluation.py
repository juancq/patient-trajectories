import numpy as np

import copy
import sys
import os
import argparse
import pickle
import time
from collections import defaultdict, OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type
from datetime import datetime
from tqdm import tqdm

import yaml
from loguru import logger

from rich.progress import track

import polars as pl

import torch
import torch.nn.functional as F

from numpy.typing import ArrayLike
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibrationDisplay

from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

import utils.predict_utils as ut
from evaluation.metrics import calculate_metrics

from halo.config_64 import HALOConfig
from halo.model import HALOModel
from halo.fine_tune_data_module_simple import HALODataModule, collate_fn
from utils.fine_tune_utils import FineTunedHALO, setup_lora

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


def validate(model, dataset, device='cuda',
    return_pred=False):
    loss = 0
    preds = []
    labels = []
    model.eval()
    #for step, batch in enumerate(track(dataset)):
    for step, batch in enumerate(dataset):
        batch = {key: value.to(device) for key,value in batch.items()}
        #d = dataset.dataset.ehr_data.get_column('length_of_stay').to_numpy()
        #e = batch['labels'].cpu()
        #import code
        #code.interact(local=locals())
        with torch.no_grad():
            outputs = model(**batch)
        loss += outputs.loss.item()
        preds.append(outputs.logits.cpu())
        labels.append(batch['labels'].cpu())

    y_pred = F.sigmoid(torch.cat(preds, dim=0)).numpy()
    y_test = torch.cat(labels, dim=0).numpy()
    metrics = calculate_metrics(y_test, y_pred, threshold=0.5)
    loss = loss / len(dataset)

    result =  {'metrics': metrics, 'loss': loss}
    if return_pred:
        result['y_pred'] = y_pred
        result['y_test'] = y_test

    return result


def eval_cv_model(
    config, 
    df: pl.DataFrame,
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
    df : Polars dataframe
        Features and labels.
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

    scores = defaultdict(lambda: defaultdict(list))
    predictions = []
    labels = []

    # Load pre-trained HALO model
    base_model = HALOModel(config)
    config.batch_size = 256
    model_name = 'halo'

    # Create fine-tuned model
    model = FineTunedHALO(base_model, config.n_embd)
    model = setup_lora(model)
    print(model)

    y = df.select(pl.col(label))
    for i, (train_index,test_index) in enumerate(tqdm(fold_indices)): 
        model.load_state_dict(torch.load(f'finetune_logs/fold_{i}/{label}_{window}_finetuned_lora.pth'))
        x_train, x_test = df[train_index], df[test_index]

        val_size = int(0.1 * len(x_train))
        train_size = len(x_train) - val_size

        x_val = x_train.slice(train_size, val_size)
        x_train = x_train.slice(0, train_size)

        data_module = HALODataModule(
                        config, 
                        batch_size=config.batch_size,
                        train_data=x_train,
                        val_data=x_val,
                        test_data=x_test,
                        task=label,
                        shuffle=False,
        )
        data_module.setup()
        test_dataset = data_module.test_dataloader()

        device = 'cuda'
        model.to(device)

        # test
        test_result = validate(model, test_dataset, return_pred=True)
        metrics, test_loss = test_result['metrics'], test_result['loss']
        print(f'best weights {metrics}, test-loss: {test_loss:.3}')

        for metric_name, metric_value in metrics.items():
            scores['full_test'][metric_name].append(metric_value)

        evaluate_subsets(
            subset_data, 
            y_test=test_result['y_test'], 
            y_pred=test_result['y_pred'], 
            test_index=test_index, 
            scores=scores
        )

        #predictions.append(test_result['y_pred'])
        #labels.append(test_result['y_test'])

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

    cv_end = time.time()
    cv_runtime = cv_end - cv_start
    print(f'CV Execution time {cv_runtime:.1f} seconds or {cv_runtime/60:.1f} minutes')

    return {'metrics':result}

def eval_cv_model_dup(
    config, 
    df: pl.DataFrame,
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
    df : Polars dataframe
        Features and labels.
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

    scores = defaultdict(lambda: defaultdict(list))
    scores_dup = defaultdict(lambda: defaultdict(list))
    predictions = []
    labels = []

    # Load pre-trained HALO model
    base_model = HALOModel(config)
    config.batch_size = 256
    model_name = 'halo'

    # Create fine-tuned model
    model = FineTunedHALO(base_model, config.n_embd)
    model = setup_lora(model)
    print(model)

    y = df.select(pl.col(label))
    for i, (train_index,test_index) in enumerate(tqdm(fold_indices)): 
        model.load_state_dict(torch.load(f'finetune_logs/fold_{i}/{label}_{window}_finetuned_lora.pth'))
        x_train, x_test = df[train_index], df[test_index]

        val_size = int(0.1 * len(x_train))
        train_size = len(x_train) - val_size

        x_val = x_train.slice(train_size, val_size)
        x_train = x_train.slice(0, train_size)

        data_module = HALODataModule(
                        config, 
                        batch_size=config.batch_size,
                        train_data=x_train,
                        val_data=x_val,
                        test_data=x_test,
                        task=label,
                        shuffle=False,
        )
        data_module.setup()
        test_dataset = data_module.test_dataloader()

        device = 'cuda'
        model.to(device)

        # test
        test_result = validate(model, test_dataset, return_pred=True)
        metrics, test_loss = test_result['metrics'], test_result['loss']
        print(f'best weights {metrics}, test-loss: {test_loss:.3}')

        for metric_name, metric_value in metrics.items():
            scores['full_test'][metric_name].append(metric_value)

        evaluate_subsets(
            subset_data, 
            y_test=test_result['y_test'], 
            y_pred=test_result['y_pred'], 
            test_index=test_index, 
            scores=scores
        )


        for subset_name, subset_mask in subset_data.items():
            logger.info(f'Evaluating subset {subset_name}')
            # gives me mask of elements in test split
            mask = subset_mask[test_index]
            test_indices = np.where(mask)[0]
            subset_dataset = Subset(data_module.test_dataset, test_indices)

            subset_loader = DataLoader(subset_dataset, 
                    batch_size=config.batch_size, 
                    shuffle=False, 
                    collate_fn=partial(collate_fn, mask_shift=False)
                    )
            subset_loader = data_module.get_loader('subset', subset_dataset)
            subset_result = validate(model, subset_loader)

            for metric_name, metric_value in subset_result['metrics'].items():
                scores_dup[subset_name][metric_name].append(metric_value)

        #predictions.append(test_result['y_pred'])
        #labels.append(test_result['y_test'])

        import code
        code.interact(local=locals())
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

    cv_end = time.time()
    cv_runtime = cv_end - cv_start
    print(f'CV Execution time {cv_runtime:.1f} seconds or {cv_runtime/60:.1f} minutes')

    return {'metrics':result}


def main():
    """
    Fit a model on dataset, where features consist of embeddings and label
    is time to death. 
    """
    main_start_time = time.time()
    pl.Config.set_tbl_cols(20)
    parser = argparse.ArgumentParser()
    #parser.add_argument('file', nargs='+', help='dataset', default=None, type=str)
    parser.add_argument('--cv', help='run cross-validation', default=False, action='store_true')
    parser.add_argument('--high', help='run cross-validation', default=False, action='store_true')
    parser.add_argument('--r', help='lora rank', default=8, type=int)
    parser.add_argument('--lora_alpha', help='lora rank', default=8, type=int)
    args = parser.parse_args()
    args.cv = True
    high = True
    data_split = 'held_out'

    root_path = Path('/mnt/fast_drive/h_drive/stats_scripts/pipeline')

    config = ut.load_config(root_path / 'predict_config.yaml')

    with open('config/best_models.yml', 'r') as file:
        model_list = yaml.safe_load(file)

    # Initialize data module
    data_path = Path('/mnt/data_volume/apdc/study1/preprocessed/tasks')

    model_settings = model_list['original']
    torch.set_float32_matmul_precision('high')
    # Load configuration
    dv = pl.read_excel("df_vars.xlsx")
    halo_config = HALOConfig(dv, **model_settings['model'])

    halo_config.model_settings = model_settings

    baseline_features = pl.scan_parquet(data_path / 'baseline_features_strong_two_year.parquet')
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

    cohort_file_name = data_path / f'df_{data_split}_halo_two_year_v2.parquet'
    raw_data = pl.scan_parquet(cohort_file_name)

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
        for window in config.windows:
            logger.info(f'Window: {window}')
            print(f'Window: {window}')

            # read label file
            labels = pl.scan_parquet(data_path / f'{task}_{window}.parquet')
             # keep only ppns present in all horizons
            labels = ids.join(labels, on='ppn_int', how='inner')
            labels = labels.sort('ppn_int')
            labels = labels.collect(streaming=streaming_flag)
            label_ids = labels.lazy().select('ppn_int')

            # convert task to binary
            labels = ut.threshold_label(labels, task, high)
            labels = labels.sort('ppn_int')

            #labels = labels.select(pl.col(['ppn_int', task]))
            window_data = labels.lazy().join(raw_data, on='ppn_int', how='inner', coalesce=True)
            window_data = window_data.sort('ppn_int')
            window_data = window_data.collect(streaming=True)

            # baseline features
            df_baseline_window = labels.lazy().join(baseline_features, on='ppn_int', coalesce=True)
            df_baseline_window = df_baseline_window.sort('ppn_int')
            df_baseline_window = df_baseline_window.collect(streaming=streaming_flag)
            subset_data = ut.build_subsets(df_baseline_window, task)

            logger.info(f'Cohort file: {cohort_file_name.name}.')

            ut.verify_data_alignment(
                {'baseline': df_baseline_window,
                'data': window_data,
                }
                , 
                labels
            )

            window_results = []
            if args.cv:
                X = df_baseline_window.with_row_index()
                y = df_baseline_window.get_column(task)
                fold_indices = list(kf.split(X, y))

                #----------------------#
                start_time = time.time()
                embedding_name = 'halo_finetune'
                embed_model_result = eval_cv_model(halo_config, window_data, 
                            fold_indices, 
                            window,
                            feature_set=embedding_name, label=task,
                            subset_data=subset_data
                )
                end_time = time.time()
                runtime = end_time - start_time
                print(f'Execution time of task {task} window {window}: {runtime:.1f} seconds or {runtime/60:.1f} minutes')

                #pred_name = f'plots/{task}_{window}_{embedding_name}_predictions_{config.run_label}.csv'
                #df_pred = pd.DataFrame(embed_model_result['predictions'])
                #df_pred.to_csv(pred_name)

                #plt.grid(True)
                ## saves calibration plot
                #plot_name = f'plots/{task}_{window}_calibration_{config.run_label}.png'
                #plt.savefig(plot_name, bbox_inches='tight')

                window_results = embed_model_result['metrics']

            formatted_df = ut.format_results(window_results)

            formatted_df['window'] = window
            results_df_all.append(formatted_df)
        ut.save_results(results_df_all, task, high, config)
    main_end_time = time.time()
    runtime = main_end_time - main_start_time
    print(f'Execution time of script: {runtime/60:.1f} minutes')


if __name__ == "__main__":
    main()
