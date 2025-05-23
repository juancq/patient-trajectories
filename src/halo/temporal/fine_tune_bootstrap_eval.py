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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

import predict_utils as ut
from metrics import calculate_metrics

from config import HALOConfig
from halo import HALOModel
from fine_tune_data_module_simple import HALODataModule, collate_fn
from fine_tune_utils import FineTunedHALO, setup_lora

streaming_flag = False

def enable_dropout_layers(model):
    """
    Activates dropout layers in a model.
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def predict(model, dataset, device='cuda'):
    loss = 0
    preds = []
    labels = []
    model.eval()
    for step, batch in enumerate(track(dataset)):
        batch = {key: value.to(device) for key,value in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss += outputs.loss.item()
        preds.append(outputs.logits.cpu())
        labels.append(batch['labels'].cpu())

    y_pred = F.sigmoid(torch.cat(preds, dim=0)).numpy()
    y_test = torch.cat(labels, dim=0).numpy()
    return {'y_test':y_test, 'y_pred': y_pred}

def bootstrap_test_set(bootstrap_indexes, y_test, y_pred, scores):
    logger.info(f'Performing {len(bootstrap_indexes)} bootstraps on test set')
    for bootstrap_i in tqdm(bootstrap_indexes):
        bootstrap_y_test = y_test[bootstrap_i]
        bootstrap_y_pred = y_pred[bootstrap_i]

        metrics = calculate_metrics(bootstrap_y_test, bootstrap_y_pred, threshold=0.5)
        for metric_name, metric_value in metrics.items():
            scores['full_test'][metric_name].append(metric_value)


def validate(model, dataset, device='cuda',
    return_pred=False):
    result = predict(model, dataset, device)
    y_test = result['y_test']
    y_pred = result['y_pred']
    metrics = calculate_metrics(y_test, y_pred, threshold=0.5)

    result =  {'metrics': metrics}
    if return_pred:
        result['y_pred'] = y_pred
        result['y_test'] = y_test

    return result


def monte_carlo_dropout(model, test_dataset):
    num_monte_carlo_passes = 5
    logger.info(f'Monte Carlo Droupout Passes: {num_monte_carlo_passes}')
    model.eval()
    with torch.no_grad():
        enable_dropout_layers(model)
        for _ in range(num_monte_carlo_passes):
            test_result = validate(model, test_dataset, return_pred=True)
            metrics, _ = test_result['metrics'], test_result['loss']

            for metric_name, metric_value in metrics.items():
                scores['full_test'][metric_name].append(metric_value)

            for subset_name, subset_mask in subset_data.items():
                logger.info(f'Evaluating subset {subset_name}')
                # gives me mask of elements in test split
                test_indices = np.where(subset_mask)[0]
                subset_dataset = Subset(data_module.test_dataset, test_indices)

                subset_loader = DataLoader(subset_dataset, 
                        batch_size=config.batch_size, 
                        shuffle=False, 
                        collate_fn=partial(collate_fn, mask_shift=False)
                        )
                subset_loader = data_module.get_loader('subset', subset_dataset)
                subset_result = validate(model, subset_loader)

                for metric_name, metric_value in subset_result['metrics'].items():
                    scores[subset_name][metric_name].append(metric_value)



def eval_cv_model(
    config, 
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
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

    scores = defaultdict(lambda: defaultdict(list))

    # Load pre-trained HALO model
    base_model = HALOModel(config)
    config.batch_size = 256
    model_name = 'halo'

    # Create fine-tuned model
    model = FineTunedHALO(base_model, config.n_embd)
    model = setup_lora(model)
    print(model)
    model.load_state_dict(torch.load(f'finetuning/{label}_{window}_finetuned_lora.pth'))

    val_size = int(0.1 * len(df_train))
    train_size = len(df_train) - val_size

    x_val = df_train.slice(train_size, val_size)
    x_train = df_train.slice(0, train_size)
    logger.info(f'val size {len(x_val)}, train size {len(x_train)}')

    data_module = HALODataModule(
                    config, 
                    batch_size=config.batch_size,
                    train_data=x_train,
                    val_data=x_val,
                    test_data=df_test,
                    task=label,
    )
    data_module.setup()
    test_dataset = data_module.test_dataloader()

    device = 'cuda'
    model.to(device)

    model.eval()
    num_points = len(test_dataset.dataset)
    n_bootstraps = 100
    bootstrap_indexes = np.random.randint(
        low=0, high=num_points, size=(n_bootstraps, num_points)
    )
    with torch.no_grad():
        model_predictions = predict(model, test_dataset)

    y_test = model_predictions['y_test']
    y_pred = model_predictions['y_pred']
    df_predictions = pl.DataFrame(
        {
            'prediction':y_pred, 
            'true':y_test,
        }
    )
    df_predictions.write_csv(f'temporal_halo_finetune_{label}_{window}_predictions.csv')

    logger.info(f'Bootstrapping on test set: {n_bootstraps}')
    bootstrap_test_set(
        bootstrap_indexes, y_test, y_pred, scores
    )


    for subset_name, subset_scores in scores.items():
        # Compile results
        mc_pass_results = {
            'subset': subset_name,
            'task':label,
            'feature_set':feature_set,
            'model':model_name,
        }

        # add mean and std for each metric
        for metric_name, metric_values in subset_scores.items():
            if metric_name == 'dcal_pvalue':
                mc_pass_results[metric_name] = np.sum(metric_values)
                mc_pass_results[f'{metric_name} std'] = 0
            else:
                mc_pass_results[metric_name] = np.mean(metric_values)
                mc_pass_results[f'{metric_name} std'] = np.std(metric_values)
                lower = np.percentile(metric_values, 2.5)
                higher = np.percentile(metric_values, 97.5)
                mc_pass_results[f'{metric_name}_CI'] = f'({lower:0.3f}-{higher:0.3f})'
            mc_pass_results[f'{metric_name}_range'] = metric_values
        
        result.append(mc_pass_results)

    cv_end = time.time()
    cv_runtime = cv_end - cv_start
    print(f'CV Execution time {cv_runtime:.1f} seconds or {cv_runtime/60:.1f} minutes')

    return {'metrics':result}


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


def main():
    """
    Fit a model on dataset, where features consist of embeddings and label
    is time to death. 
    """
    main_start_time = time.time()
    pl.Config.set_tbl_cols(20)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--train', type=str, default=None)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--high', help='run cross-validation', default=False, action='store_true')
    parser.add_argument('--r', help='lora rank', default=8, type=int)
    parser.add_argument('--lora_alpha', help='lora rank', default=8, type=int)
    args = parser.parse_args()
    high = True

    root_path = Path('/mnt/fast_drive/h_drive/stats_scripts/pipeline')

    config = ut.load_config(root_path / 'temporal/temporal_eval_config.yaml')

    with open('best_models.yml', 'r') as file:
        model_list = yaml.safe_load(file)

    # Initialize data module
    data_path = Path('/mnt/data_volume/apdc/study1/preprocessed/tasks')

    model_name = args.model
    if model_name not in model_list:
        logger.error('Model name not found in model yaml. Add model name and architecture details to best_models.yaml')
    model_settings = model_list[model_name]

    torch.set_float32_matmul_precision('high')
    # Load configuration
    dv = pl.read_excel("temporal_df_vars.xlsx")
    halo_config = HALOConfig(dv, **model_settings['model'])

    halo_config.model_settings = model_settings

    train_data = pl.scan_parquet(Path(args.train))
    test_data = pl.scan_parquet(Path(args.test))

    logger.info('Getting baseline train features')
    baseline_features_train = get_baseline_features(
        config['baseline']['path'],
        is_train=True,
        config=config['baseline']
    )

    logger.info('Getting baseline test features')
    baseline_features_test = get_baseline_features(
        config['baseline']['path'],
        is_train=False,
        config=config['baseline']
    )

    baseline_features_default = {
        'train': baseline_features_train,
        'test': baseline_features_test,
    }
    #import code
    #code.interact(local=locals())

    # process each task
    for task in config.tasks.keys():
        logger.info(f'Task: {task}')
        results_df_all = []

        # process each prediction horizon
        #for window in config.windows:
        for window in ['six_months']:
            logger.info(f'Window: {window}')
            print(f'Window: {window}')

            # read label file
            labels_train = pl.scan_parquet(data_path / f'temporal_{task}_{window}_train.parquet')
            labels_test = pl.scan_parquet(data_path / f'temporal_{task}_{window}_test.parquet')
            #labels_train = labels_train.head(50000)
            #labels_test = labels_test.head(50000)
            labels_train = labels_train.collect(streaming=True)
            labels_test = labels_test.collect(streaming=True)

            labels_train = labels_train.sample(
                fraction=1.0, 
                with_replacement=False, 
                shuffle=True,
                seed=123,
            )
            labels_test = labels_test.sample(
                fraction=1.0, 
                with_replacement=False, 
                shuffle=True,
                seed=123,
            )

            # convert task to binary
            labels_train, task_threshold = ut.threshold_label(labels_train, task, high=high, threshold=None)

            labels_test, _ = ut.threshold_label(labels_test, task, high=high, threshold=task_threshold)

            window_train_data = labels_train.join(train_data.collect(), on='ppn_int')
            window_test_data = labels_test.join(test_data.collect(), on='ppn_int')

            baseline_features_test_with_label = baseline_features_test.join(labels_test.lazy(), on='ppn_int').collect()
            subset_data = ut.build_subsets(baseline_features_test_with_label, task)

            start_time = time.time()
            #halo_config.lora_r = args.r
            embedding_name = 'halo_finetune'
            embed_model_result = eval_cv_model(
                halo_config, 
                window_train_data, 
                window_test_data, 
                window,
                feature_set=embedding_name, 
                label=task,
                subset_data=subset_data
            )

            end_time = time.time()
            runtime = end_time - start_time
            print(f'Execution time of task {task} window {window}: {runtime:.1f} seconds or {runtime/60:.1f} minutes')

            window_results = embed_model_result['metrics']

            formatted_df = ut.format_results(window_results)

            formatted_df['window'] = window
            results_df_all.append(formatted_df)
            ut.save_results(results_df_all, task, high, config, temp='temp')

        ut.save_results(results_df_all, task, high, config)

    main_end_time = time.time()
    runtime = main_end_time - main_start_time
    print(f'Execution time of script: {runtime/60:.1f} minutes')


if __name__ == "__main__":
    main()
