from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch3d.transforms import *
from data import TeethDataset

import pytorch_lightning as pl

class DataAugmentation(nn.Module):
    def __init__(self):
        super(DataAugmentation, self).__init__()
        self.transforms = Transform3d()

    @torch.no_grad()
    def forward(self, X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        teeth_num = X["X_v"].shape[0]
        trans = self.transforms.compose(Rotate(euler_angles_to_matrix(torch.randint(-30, 30, (28, 3)), "XYZ")),
                                        Translate(torch.randn(teeth_num, 3)),
                                        )
        X["X_v"] = trans.transform_points(X["X_v"])
        X["X"] = X["X_v"].reshape(shape=X["X"].shape)
        X["6dof"] = se3_log_map(trans.inverse().get_matrix())
        return X

class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        num_workers: int,
        split_ratio: float = 0.25,
    ):
        super(LitDataModule, self).__init__()

        self.data_root = data_root

        self.transform = DataAugmentation()

        self.teeth_dataset = TeethDataset(self.data_root,  transform=self.transform)

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        teeth_train, teeth_test = random_split(self.teeth_dataset, [len(self.teeth_dataset)-(int)(len(self.teeth_dataset)*self.hparams.split_ratio), (int)(len(self.teeth_dataset)*self.hparams.split_ratio)])
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = random_split(teeth_train, [len(teeth_train)-(int)(len(teeth_train)*0.15), (int)(len(teeth_train)*0.15)])
        if stage == "test" or stage is None:
            self.test_dataset = teeth_test
            
    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)
    
    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: TeethDataset, train: bool = False) -> DataLoader:
        return DataLoader(dataset,
                        batch_size=self.hparams.batch_size,
                        shuffle=train,
                        num_workers=self.hparams.num_workers,
                        pin_memory=True,
                        drop_last=train,
                        )
