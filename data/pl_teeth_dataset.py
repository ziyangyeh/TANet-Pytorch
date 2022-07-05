from posixpath import split
from typing import Dict, List, Optional, Tuple, Callable
import pandas as pd
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
        teeth_num = X["C"].shape[0]

        X["X_v"] = Translate(-X["C"]).transform_points(X["X_v"])
        trans = self.transforms.compose(Rotate(euler_angles_to_matrix(torch.randint(-30, 30, (28, 3)), "XYZ")),
                                        Translate(torch.randn(teeth_num, 3)),
                                        )
        X["X_v"] = trans.transform_points(X["X_v"])
        X["X_v"] = Translate(X["C"]).transform_points(X["X_v"])
        X["X"] = X["X_v"].clone().reshape(shape=X["X"].shape)
        X_matrices = trans.inverse().get_matrix()

        teeth_num = X["target_X_v"].shape[0]
        X["target_X_v"] = Translate(-X["C"]).transform_points(X["target_X_v"])
        trans = self.transforms.compose(Rotate(euler_angles_to_matrix(torch.randint(-30, 30, (28, 3)), "XYZ")),
                                        Translate(torch.randn(teeth_num, 3)),
                                        )
        X["target_X_v"] = trans.transform_points(X["target_X_v"])
        X["target_X_v"] = Translate(X["C"]).transform_points(X["target_X_v"])
        X["target_X"] = X["target_X_v"].clone().reshape(shape=X["target_X"].shape)
        target_X_matrices = trans.get_matrix()
        
        trans = self.transforms.compose(Rotate(X_matrices[:, :3, :3]),
                                        Rotate(target_X_matrices[:, :3, :3]),
                                        )
        final_trans_mat = trans.get_matrix()
        final_trans_mat[:, 3, :3] = X_matrices[:, 3, :3].add(target_X_matrices[:, 3, :3])
        
        X["6dof"] = se3_log_map(final_trans_mat)
        return X

class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_csv: str,
        batch_size: int,
        num_workers: int,
        split_ratio: float = 0.25,
    ):
        super(LitDataModule, self).__init__()

        self.data_set = pd.read_csv(data_csv)

        self.split_ratio = split_ratio

        self.train_val_df = self.data_set.sample(frac=1-self.split_ratio)

        self.test_df = self.data_set.drop(self.train_val_df.index)

        self.transform = DataAugmentation()

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        
        if stage == "fit" or stage is None:
            train_df = self.train_val_df.sample(frac=1-self.split_ratio)
            val_df = self.train_val_df.drop(train_df.index)
            self.train_dataset = TeethDataset(train_df, transform=self.transform)
            self.val_dataset = TeethDataset(val_df, transform=self.transform)
        if stage == "test" or stage is None:
            self.test_dataset = TeethDataset(self.test_df)
            
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
