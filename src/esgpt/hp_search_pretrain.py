#!/usr/bin/env python
"""Pre-trains a model from scartch."""

try:
    # This color-codes and prettifies error messages if the script fails.
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import copy
import hydra
import mlflow
import optuna
import os
import torch
from functools import partial

from omegaconf import OmegaConf

from EventStream.logger import hydra_loguru_init
from EventStream.transformer.lightning_modules.optuna_trial_pretrain import (
    PretrainConfig,
    train_trial,
)

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_name="pretrain_config")
def main(cfg: PretrainConfig):
    hydra_loguru_init()
    '''
    print(cfg, type(cfg))

    print('main num hidden layers', cfg['config']['num_hidden_layers'])

    if type(cfg) is not PretrainConfig:
        cfg_object = hydra.utils.instantiate(cfg, _convert_="object")
        print('hello world#############')
        print(cfg_object.config)
    cfg['config']['num_hidden_layers']=111
    print('befor e instantiation')
    cfg_object.config = hydra.utils.instantiate(cfg.config, _convert_="object")
    print('after instantiation')
    print(cfg_object.config)
    print('seq attention layers', cfg_object.config.seq_attention_layers)
    print('num hidden layers', cfg_object.config.num_hidden_layers)
    print('end')
    return
    '''

    #print(type(cfg), cfg)
    #print(cfg.do_use_filesystem_sharing)
    #print(cfg.config.num_attention_heads)
    #print(cfg.config['num_attention_heads'])

    objective_wrapper = partial(train_trial, cfg=cfg)

    mlflow.set_tracking_uri('file:./mlruns')

    with mlflow.start_run(run_name='hpsearch_val', nested=True):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_wrapper, n_trials=20)

        mlflow.log_params(study.best_params)
        mlflow.log_metric('best_loss', study.best_value)

        mlflow.set_tags(
            tags= {
                'project': 'esgpt',
                'blah': 3
                }
        )

    #best_trial = study.best_trial
    #print(f'best_trial = {best_trial.params}')


if __name__ == "__main__":
    main()
