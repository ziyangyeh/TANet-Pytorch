import sys, os
sys.path.append(os.getcwd())

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models import LitModule
from data import LitDataModule
from openpoints.utils import EasyConfig

def train(cfg_path: str,
        dataset_csv_path: str,
        checkpoints_dir: str,
        accumulate_grad_batches: int = 1,
        auto_lr_find: bool = False,
        auto_scale_batch_size: bool = False,
        fast_dev_run: bool = False,
        gpus: int = 1,
        ):
    pl.seed_everything(42)
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)
    cfg = cfg.train
    max_epochs=cfg.epochs

    wandb_logger = WandbLogger(project="TANet")

    datamodule = LitDataModule(dataset_csv_path, cfg.batch_size, cfg.num_workers, cfg.sample_num, cfg.split_ratio)

    datamodule.setup()

    module = LitModule(cfg_path=cfg_path)

    model_checkpoint = ModelCheckpoint(checkpoints_dir,
                                        monitor="val_loss",
                                        filename=f"{module.model.__class__.__name__}_{module.model.global_encoder.__class__.__name__}",
                                        )

    trainer = pl.Trainer(accumulate_grad_batches=accumulate_grad_batches,
                        auto_lr_find=auto_lr_find,
                        auto_scale_batch_size=auto_scale_batch_size,
                        benchmark=True,
                        callbacks=[model_checkpoint],
                        deterministic=True,
                        fast_dev_run=fast_dev_run,
                        gpus=gpus,
                        max_epochs=max_epochs,
                        precision=cfg.precision,
                        log_every_n_steps=5,
                        logger=wandb_logger
                        )

    trainer.tune(module, datamodule=datamodule)

    trainer.fit(module, datamodule=datamodule)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("-c", "--config_file", type=str, metavar="", help="configuration file", default="config/default.yaml")
    parser.add_argument("-d", "--dataset_csv_dir", type=str, metavar="", help="dateset csv directory", default="dataset_csv")
    parser.add_argument("-s", "--checkpoint_dir", type=str, metavar="", help="checkpoint directory", default="tmp/checkpoint")

    args = parser.parse_args()


    train(args.config_file, args.dataset_csv_dir, args.checkpoint_dir)
