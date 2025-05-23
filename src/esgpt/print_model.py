#!/usr/bin/env python
"""Prints a model using a given pretrain config file"""

import os
import hydra

from torchinfo import summary
from torchview import draw_graph

from EventStream.transformer.lightning_modules.generative_modeling import (
    PretrainConfig,
    ESTForGenerativeSequenceModelingLM
)

from EventStream.data.pytorch_dataset import PytorchDataset


@hydra.main(version_base=None, config_name="pretrain_config")
def main(cfg: PretrainConfig):
    if type(cfg) is not PretrainConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")

    train_pyd = PytorchDataset(cfg.data_config, split="train")

    cfg.config.set_to_dataset(train_pyd)

    # Model
    LM = ESTForGenerativeSequenceModelingLM(
        config=cfg.config,
        optimization_config=cfg.optimization_config,
        metrics_config=cfg.pretraining_metrics_config,
    )

    print(LM)

    summary(LM, verbose=1)
    summary(LM, verbose=2)

    #model_graph = draw_graph(LM, save_graph=True, filename='model.dot')



if __name__ == "__main__":
    main()
