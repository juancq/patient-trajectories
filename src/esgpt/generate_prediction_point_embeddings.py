import copy
import glob
import hydra
import os
import torch
import lightning as L
import polars as pl

from loguru import logger
from pathlib import Path

from EventStream.transformer.lightning_modules.estforembedding import (
    ESTForEmbedding
)

from EventStream.transformer.lightning_modules.generative_modeling import (
    PretrainConfig,
    ESTForGenerativeSequenceModelingLM
)

from EventStream.data.pytorch_dataset import PytorchDataset


streaming_flag = False

@hydra.main(version_base=None, config_name="pretrain_config")
def main(cfg: PretrainConfig):
    print(type(cfg))
    if not isinstance(cfg, PretrainConfig):
        cfg = hydra.utils.instantiate(cfg, _convert_="object")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        cfg_dict = copy.deepcopy(cfg)
        cfg_dict.config = cfg_dict.config.to_dict()

    pooling_method = cfg.config.task_specific_params['pooling_method']
    #L.seed_everything(cfg.seed)

    # for overwriting
    #cfg.load_from_checkpoint = ''

    torch.set_float32_matmul_precision('high')
    optimization_config = cfg.optimization_config

    # death 
    # these embeddings are task invariant for the purpose of study 1
    # we just want the embeddings based on the calculated prediction point
    task_name = cfg.data_config.task_df_name 
    cfg.data_config.task_df_name  = None
    # we load the train dataset only to set the 
    train_pyd = PytorchDataset(cfg.data_config, split="train")
    cfg.config.set_to_dataset(train_pyd)
    optimization_config.set_to_dataset(train_pyd)
    cfg.data_config.task_df_name  = task_name
    #tuning_pyd = PytorchDataset(cfg.data_config, split="tuning")
    held_out_pyd = PytorchDataset(cfg.data_config, split="held_out")

    # using this dataframe to add ppn_int to the generated embeddings, as subject_id != ppn_int
    subjects_df = pl.scan_parquet(Path(cfg.data_config.save_dir) / 'subjects_df.parquet')
    # have to cast ppn_int as string as it is categorical variable, and without casting you get
    # the categorical value and not the ppn
    subjects_df = subjects_df.select(pl.col('subject_id'))

    if cfg.load_from_checkpoint is None: 
        logger.info(f'Attempting to load best model from {cfg.load_from_model_dir}')
        checkpoints = glob.glob(cfg.load_from_model_dir + '/lightning_logs/epoch*.ckpt')
        values = [float(ckpt.split('tuning_loss=')[-1].split('.ckpt')[0]) for ckpt in checkpoints]
        min_index = values.index(min(values))

        cfg.load_from_checkpoint = checkpoints[min_index]

    logger.info(f'Using {cfg.load_from_checkpoint} as best model to generate embeddings')

    checkpoint = ESTForGenerativeSequenceModelingLM.load_from_checkpoint(cfg.load_from_checkpoint)

    # Setting up dataloader, we just want the embeddings for 
    # tuning (used for experimentation of downstream tasks) and held-out (the actual dataset
    # used for generating downstream task results
    #tuning_dataloader = torch.utils.data.DataLoader(
    #    tuning_pyd,
    #    batch_size=optimization_config.validation_batch_size,
    #    num_workers=optimization_config.num_dataloader_workers,
    #    collate_fn=tuning_pyd.collate,
    #    shuffle=False,
    #)

    held_out_dataloader = torch.utils.data.DataLoader(
        held_out_pyd,
        batch_size=optimization_config.validation_batch_size,
        num_workers=optimization_config.num_dataloader_workers,
        collate_fn=held_out_pyd.collate,
        shuffle=False,
    )

    cfg.trainer_config['enable_checkpointing'] = False
    cfg.trainer_config['logger'] = False

    trainer = L.Trainer(**cfg.trainer_config)

    model = ESTForEmbedding(pooling_method=pooling_method, model=checkpoint.model.encoder)
    for sp, dataloader in (
        #("tuning", tuning_dataloader),
        ("held_out", held_out_dataloader),
    ):
        embeddings_fp = Path(cfg.load_from_model_dir) / "embeddings" / f"{sp}_{task_name}_embeddings_{pooling_method}.pt"
        if os.environ.get("LOCAL_RANK", "0") == "0":
            if embeddings_fp.is_file() and not cfg.do_overwrite:
                logger.debug(f"Embeddings already exist at {embeddings_fp}. To overwrite, set `do_overwrite=True`.")
                continue

        embeddings = torch.cat(trainer.predict(model, dataloader), 0)

        embed_df = pl.LazyFrame(embeddings.numpy(), 
                schema=[f'feature_{i+1}' for i in range(embeddings.shape[1])])

        # add subject_id to embeddings, subject_id is esgpt dependent
        #embed_df = embed_df.with_columns(pl.Series('ppn_int', dataloader.dataset.subject_ids))

        embed_df = embed_df.with_columns(pl.Series('ppn_int', dataloader.dataset.subject_ids))


        # what is the alternative to this?
        if os.environ.get("LOCAL_RANK", "0") == "0":
            logger.info(f"Saving {sp} embeddings to {embeddings_fp}.")
            embeddings_fp.parent.mkdir(exist_ok=True, parents=True)
            embed_df.collect(streaming=streaming_flag).write_parquet(embeddings_fp, use_pyarrow=True)
            current_folder = f"embeddings/{Path(cfg.experiment_dir).stem}_{sp}_{task_name}_embeddings_{pooling_method}.pt"
            embed_df.collect(streaming=streaming_flag).write_parquet(current_folder, use_pyarrow=True)



if __name__ == "__main__":
    main()
