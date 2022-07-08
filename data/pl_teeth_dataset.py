from typing import Dict, List, Optional, Tuple, Callable
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch3d.transforms import *

from data import TeethDataset

import pytorch_lightning as pl

class TrainDataAugmentation(nn.Module):
    def __init__(self):
        super(TrainDataAugmentation, self).__init__()

    @torch.no_grad()
    def forward(self, X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        teeth_num = X["C"].shape[0]

        trans = Transform3d().compose(Translate(-X["C"]),
                                    Rotate(euler_angles_to_matrix(torch.randint(-30, 30, (teeth_num, 3)), "XYZ")),
                                    Translate(torch.randn(teeth_num, 3)),
                                    Translate(X["C"]))
        X["X_v"] = trans.transform_points(X["X_v"])
        X["X"] = X["X_v"].clone().reshape(shape=X["X"].shape)
        X_matrices = trans.inverse().get_matrix()

        trans = Transform3d().compose(Translate(-X["C"]),
                                    Rotate(euler_angles_to_matrix(torch.randint(-30, 30, (teeth_num, 3)), "XYZ")),
                                    Translate(torch.randn(teeth_num, 3)),
                                    Translate(X["C"]))
        X["target_X_v"] = trans.transform_points(X["target_X_v"])
        X["target_X"] = X["target_X_v"].clone().reshape(shape=X["target_X"].shape)
        target_X_matrices = trans.get_matrix()

        trans = Transform3d().compose(Transform3d(matrix=X_matrices),
                                    Transform3d(matrix=target_X_matrices))

        final_trans_mat = trans.get_matrix()
        X["6dof"] = se3_log_map(final_trans_mat)

        return X

class TestDataAugmentation(nn.Module):
    def __init__(self):
        super(TestDataAugmentation, self).__init__()

    @torch.no_grad()
    def forward(self, X: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        teeth_num = X["C"].shape[0]

        trans = Transform3d().compose(Translate(-X["C"]),
                                    Rotate(euler_angles_to_matrix(torch.randint(-30, 30, (teeth_num, 3)), "XYZ")),
                                    Translate(torch.randn(teeth_num, 3)),
                                    Translate(X["C"]))
        X["X_v"] = trans.transform_points(X["X_v"])
        X["X"] = X["X_v"].clone().reshape(shape=X["X"].shape)
        X_matrices = trans.inverse().get_matrix()

        trans = Transform3d().compose(Transform3d(matrix=X_matrices))

        final_trans_mat = trans.get_matrix()
        X["6dof"] = se3_log_map(final_trans_mat)

        return X

class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_csv_path: str,
        batch_size: int,
        num_workers: int,
        sample_num: int = 512,
        split_ratio: float = 0.25,
    ):
        super(LitDataModule, self).__init__()

        self.dataset_csv_path = dataset_csv_path

        self.data_set = pd.read_csv(os.path.join(self.dataset_csv_path, "data_set.csv"))

        self.sample_num = sample_num

        self.split_ratio = split_ratio

        if os.path.exists(os.path.join(self.dataset_csv_path, "trian_val.csv")):
            self.train_val_df = pd.read_csv(os.path.join(self.dataset_csv_path, "trian_val.csv"))
        else:
            self.train_val_df = self.data_set.sample(frac=1-self.split_ratio)
            self.train_val_df.to_csv(os.path.join(self.dataset_csv_path,"trian_val.csv"), index=False)

        if os.path.exists(os.path.join(self.dataset_csv_path, "test.csv")):
            self.test_df = pd.read_csv(os.path.join(self.dataset_csv_path, "test.csv"))
        else:
            self.test_df = self.data_set.drop(self.train_val_df.index)
            self.test_df.to_csv(os.path.join(self.dataset_csv_path,"test.csv"), index=False)

        self.train_transform = TrainDataAugmentation()
        self.test_transform = TestDataAugmentation()

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            if os.path.exists(os.path.join(self.dataset_csv_path, "train.csv")) and os.path.exists(os.path.join(self.dataset_csv_path, "val.csv")):
                train_df = pd.read_csv(os.path.join(self.dataset_csv_path, "train.csv"))
                val_df = pd.read_csv(os.path.join(self.dataset_csv_path, "val.csv"))
            else:
                train_df = self.train_val_df.sample(frac=1-self.split_ratio)
                val_df = self.train_val_df.drop(train_df.index)
                train_df.to_csv(os.path.join(self.dataset_csv_path,"train.csv"), index=False)
                val_df.to_csv(os.path.join(self.dataset_csv_path,"val.csv"), index=False)
            self.train_dataset = TeethDataset(train_df, self.sample_num, transform=self.train_transform)
            self.val_dataset = TeethDataset(val_df, self.sample_num, transform=self.train_transform)
        if stage == "test" or stage is None:
            self.test_dataset = TeethDataset(self.test_df, self.sample_num, transform=self.test_transform)
            
    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)
    
    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, train=True)

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
