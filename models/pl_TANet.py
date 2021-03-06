from typing import Dict, List, Optional, Tuple, Callable
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pl_bolts.optimizers import lr_scheduler

from models.submodules import Tooth_Assembler, Tooth_Centering
from losses import ConditionalWeightingLoss
from openpoints.utils import EasyConfig
from openpoints.models.build import build_model_from_cfg

class LitModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
    ):
        super(LitModule, self).__init__()

        self.cfg = cfg
        self.cfg_optimizer = self.cfg.LitModule.optimizer
        self.cfg_scheduler = self.cfg.LitModule.scheduler
        self.cfg_scheduler.epochs = self.cfg.Trainer.epochs
        self.batch_size = self.cfg.LitDataModule.dataloader.batch_size
        self.learning_rate = self.cfg.LitModule.optimizer.learning_rate

        self.tooth_centering = Tooth_Centering()
        self.model = build_model_from_cfg(self.cfg.model)
        self.tooth_assembler = Tooth_Assembler()

        self.loss_fn = ConditionalWeightingLoss(sigma=5, criterion_mode=cfg.criterion.mode)

    def forward(self, X: Dict[str, torch.Tensor]) -> torch.Tensor:
        dof = self.model(X)
        return dof

    def configure_optimizers(self):
        # Setup the optimizer
        optimizer = create_optimizer_v2(self.parameters(),
                                        opt=self.cfg_optimizer.NAME,
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
        center_batch = self.tooth_centering(batch, self.device)
        dofs, Xi = self(center_batch)
        assembled, pred2gt_matrices = self.tooth_assembler(batch, dofs, self.device)
        loss = self.loss_fn(assembled, batch["target_X_v"], pred2gt_matrices, batch["C"].shape[1], Xi, self.device)

        self.log(f"{step}_loss", loss)

        return loss
