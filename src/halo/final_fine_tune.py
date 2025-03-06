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


def validate(model, dataset, device='cuda',
    return_pred=False):
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
    save_dir = Path(f'folds/{label}/{window}/')
    # for the calibration plot
    #fig, ax = plt.subplots(figsize=(12,6))

    #scores = defaultdict(list)
    scores = defaultdict(lambda: defaultdict(list))
    predictions = []
    labels = []

    # Load pre-trained HALO model
    base_model = HALOModel(config)
    #checkpoint = torch.load('save/halo_model-epoch=20-val_loss=0.000_128_flash_swig.ckpt', weights_only=True)
    #checkpoint = torch.load('save/halo_model-epoch=34-val_loss=0.0002-step=1307565-original.ckpt', weights_only=True)
    #checkpoint = torch.load('save/halo_model-epoch=34-val_loss=0.0002-step=1307565-original.ckpt', weights_only=True)
    checkpoint = torch.load(config.model_settings['path'], weights_only=True)
    state_dict = {k.replace('_orig_mod.', ''):v for k,v in checkpoint['state_dict'].items()}

    config.batch_size = 256
    model_name = 'halo'

    base_model.load_state_dict(state_dict)
    # Create fine-tuned model
    model = FineTunedHALO(base_model, config.n_embd)
    model = setup_lora(model)
    print(model)
    default_weights = copy.deepcopy(model.state_dict())


    y = df.select(pl.col(label))
    for i, (train_index,test_index) in enumerate(tqdm(fold_indices)):
        x_train, x_test = df[train_index], df[test_index]
        writer = SummaryWriter(log_dir=f"finetune_logs/fold_{i}")

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
        )
        data_module.setup()
        train_dataset = data_module.train_dataloader()
        val_dataset = data_module.val_dataloader()
        test_dataset = data_module.test_dataloader()

        model.load_state_dict(default_weights)
        optimizer = AdamW(params=model.parameters(), lr=1e-4)

        num_epochs = 5

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0.05 * len(train_dataset) * num_epochs,
            #num_warmup_steps=0.3 * len(train_dataset) * num_epochs,
            num_training_steps=len(train_dataset) * num_epochs,
        )
        device = 'cuda'
        model.to(device)

        patience = 5
        best_val_loss = float('inf')
        patience_count = 0
        best_weights = copy.deepcopy(model.state_dict())
        best_weights_epoch = 0
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for step,batch in enumerate(track(train_dataset)):
            #for step,batch in enumerate(train_dataset):
                batch = {key: value.to(device) for key,value in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            train_loss = train_loss / len(train_dataset)

            # validation
            model.eval()
            val_result = validate(model, val_dataset)
            print(len(train_dataset), len(val_dataset))
            val_metrics, val_loss = val_result['metrics'], val_result['loss']
            print(f"epoch {epoch}: {val_metrics}, train-loss: {train_loss:.3} val-loss: {val_loss:.3}")

            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/validation', val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                #model.save_pretrained('./temp/best_tuned_model.pth')
                #merged = model.merge_and_unload()
                #torch.save(merged.state_dict(), f'temp/model_{epoch}_{val_loss:.3}.pth')
                best_weights = copy.deepcopy(model.state_dict())
                best_weights_epoch = epoch
                print('##### ---> best weights at', epoch)
            else:
                patience_count = patience_count + 1

            if patience_count >= patience:
                print('Early stopping')
                break
            print(f'patience count {patience_count} at epoch {epoch}')

        print('^'*30)
        print('Val at last epoch')
        print('^'*30)
        # val 1
        val_result = validate(model, val_dataset)
        metrics, val_loss = val_result['metrics'], val_result['loss']
        print(f'val at last epoch {metrics}, val-loss: {val_loss:.3}')

        print('^'*30)
        print('Test at last epoch')
        print('^'*30)
        test_result = validate(model, test_dataset)
        metrics, test_loss = test_result['metrics'], test_result['loss']
        print(f'test at last epoch {metrics}, test-loss: {test_loss:.3}')

        # val 2
        model.load_state_dict(best_weights)
        model.eval()
        print('^'*30)
        print(f'Best weights found at {best_weights_epoch}')
        print('^'*30)
        print('Val at best')
        print('^'*30)
        val_result = validate(model, val_dataset)
        metrics, val_loss = val_result['metrics'], val_result['loss']
        print(f'best weights val {metrics}, val-loss: {val_loss:.3}')

        print('^'*30)
        print('Test at best')
        print('^'*30)

        # test 2
        test_result = validate(model, test_dataset, return_pred=True)
        metrics, test_loss = test_result['metrics'], test_result['loss']
        print(f'best weights {metrics}, test-loss: {test_loss:.3}')

        for metric_name, metric_value in metrics.items():
            scores['full_test'][metric_name].append(metric_value)
        predictions.append(test_result['y_pred'])
        labels.append(test_result['y_test'])

        torch.save(model.state_dict(), f'finetune_logs/fold_{i}/{label}_{window}_finetuned_lora.pth')

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
                scores[subset_name][metric_name].append(metric_value)

        writer.close()
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
        
        result.append(fold_results)

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)

    per_model_predictions[f'{model_name}_y_pred'] = predictions
    per_model_predictions[f'{model_name}_y_true'] = labels

    cv_end = time.time()
    cv_runtime = cv_end - cv_start
    print(f'CV Execution time {cv_runtime:.1f} seconds or {cv_runtime/60:.1f} minutes')

    return {'metrics':result, 'predictions': per_model_predictions}


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
                save_indices = {
                    'fold_indices': fold_indices,
                    'subset_data': subset_data
                }
                save_dir = Path(f'folds/{task}/{window}/')
                save_dir.mkdir(parents=True, exist_ok=True)
                with open(save_dir / 'fold_indices_2.pkl', 'wb') as f:
                    pickle.dump(save_indices, f)
                save_index = []
                for train_index, test_index in fold_indices:
                    train_data = X.filter(pl.col('index').is_in(train_index)).select(pl.col(['index', 'ppn_int']))
                    test_data = X.filter(pl.col('index').is_in(test_index)).select(pl.col(['index', 'ppn_int']))
                    fold = {'train': train_data, 'test': test_data}
                    save_index.append(fold)

                with open(save_dir / 'fold_indices_ppn_finetune.pkl', 'wb') as f:
                    pickle.dump(save_index, f)

                # in dataloader, I need to join with the given labels
                # add test set to dataloader
                # train with early stopping
                # then get stats on test set
                #----------------------#

                start_time = time.time()
                #halo_config.lora_r = args.r
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
            ut.save_results(results_df_all, task, high, config, temp='temp')
        ut.save_results(results_df_all, task, high, config)
    main_end_time = time.time()
    runtime = main_end_time - main_start_time
    print(f'Execution time of script: {runtime/60:.1f} minutes')


if __name__ == "__main__":
    main()
