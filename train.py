import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from models import LitModule
from data import LitDataModule
from openpoints.utils import EasyConfig

def train(cfg_path: str,
        data_csv: str,
        checkpoints_dir: str,
        accumulate_grad_batches: int = 2,
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

    datamodule = LitDataModule(data_csv, cfg.batch_size, cfg.num_workers, cfg.split_ratio)

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
                        limit_train_batches=1.0,
                        limit_val_batches=1.0,
                        )

    trainer.tune(module, datamodule=datamodule)

    trainer.fit(module, datamodule=datamodule)

    trainer.test(module, datamodule=datamodule)

train("config/default.yaml", "data_set.csv", "tmp/checkpoint")
