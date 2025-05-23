#!/usr/bin/env python

try:
    # This color-codes and prettifies error messages if the script fails.
    import stackprinter

    stackprinter.set_excepthook(style="darkbg2")
except ImportError:
    pass  # no need to fail because of missing dev dependency

import copy
import os

import hydra


from omegaconf import OmegaConf

from EventStream.transformer.lightning_modules.save_pytorch_model_from_checkpoint import (
    PretrainConfig,
    convert_checkpoints,
)


@hydra.main(version_base=None, config_name="pretrain_config")
def main(cfg: PretrainConfig):
    if type(cfg) is not PretrainConfig:
        cfg = hydra.utils.instantiate(cfg, _convert_="object")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        cfg_dict = copy.deepcopy(cfg)
        cfg_dict.config = cfg_dict.config.to_dict()

    return convert_checkpoints(cfg, cfg_dict)


if __name__ == "__main__":
    main()
