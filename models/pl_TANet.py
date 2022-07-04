from typing import Dict, List, Optional, Tuple, Callable
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pl_bolts.optimizers import lr_scheduler

from models.submodules import Tooth_Assembler
from openpoints.utils import EasyConfig
from openpoints.models.build import build_model_from_cfg


class LitModule(pl.LightningModule):
    def __init__(
        self,
        cfg_path: str,
    ):
        super(LitModule, self).__init__()

        self.cfg = EasyConfig()
        self.cfg.load(cfg_path, recursive=True)

        self.cfg_optimizer = self.cfg.train.optimizer
        self.cfg_scheduler = self.cfg.train.scheduler
        self.cfg_scheduler.epochs = self.cfg.train.epochs

        self.save_hyperparameters()

        self.model = build_model_from_cfg(self.cfg.model)
        self.tooth_assembler = Tooth_Assembler()

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        dof = self.model(X)
        
        return dof

    def configure_optimizers(self):
        # Setup the optimizer
        optimizer = create_optimizer_v2(self.parameters(),
                                        opt=self.cfg_optimizer.optimizer,
                                        lr=self.cfg_optimizer.learning_rate,
                                        weight_decay=self.cfg_optimizer.weight_decay,
                                        )

        # Setup the schedulerwarmup_epochs: int,
        scheduler = lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer,
                                                            warmup_epochs=self.cfg_scheduler.warmup_epochs,
                                                            max_epochs=self.cfg_scheduler.epochs,
                                                            warmup_start_lr=self.cfg_scheduler.warmup_start_lr,
                                                            eta_min=self.cfg_scheduler.eta_min,
                                                            last_epoch=-1,
                                                            )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        dofs = self(batch)
        assembled = self.tooth_assembler(batch, dofs, self.device)
        return None
        print(dofs.shape)

        # loss = self.loss_fn(dof, y)
        
        # self.log(f"{step}_loss", loss)

        # return loss