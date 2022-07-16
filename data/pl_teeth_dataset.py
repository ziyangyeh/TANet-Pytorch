from typing import Dict, List, Optional, Tuple, Callable
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch3d.transforms import *

from data import TeethDataset, H5TeethDataset

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
        cfg,
    ):
        super(LitDataModule, self).__init__()
        self.cfg = cfg

        self.batch_size = cfg.dataloader.batch_size
        self.num_workers = cfg.dataloader.num_workers
        self.dataset_csv_path = cfg.dataset.csv_dir

        self.data_set = pd.read_csv(os.path.join(self.dataset_csv_path, "data_set.csv"))

        self.sample_num = cfg.dataset.sample_num

        self.val_split_ratio = cfg.dataset.val.split_ratio
        self.test_split_ratio = cfg.dataset.test.split_ratio

        if os.path.exists(os.path.join(self.dataset_csv_path, "trian_val.csv")):
            self.train_val_df = pd.read_csv(os.path.join(self.dataset_csv_path, "trian_val.csv"))
        else:
            self.train_val_df = self.data_set.sample(frac=1-self.test_split_ratio)
            self.train_val_df.to_csv(os.path.join(self.dataset_csv_path,"trian_val.csv"), index=False)

        if os.path.exists(os.path.join(self.dataset_csv_path, "test.csv")):
            self.test_df = pd.read_csv(os.path.join(self.dataset_csv_path, "test.csv"))
        else:
            self.test_df = self.data_set.drop(self.train_val_df.index)
            self.test_df.to_csv(os.path.join(self.dataset_csv_path,"test.csv"), index=False)

        self.train_transform = TrainDataAugmentation()
        self.test_transform = TestDataAugmentation()

    def enum_dataset(self):
        if os.path.exists(os.path.join(self.dataset_csv_path, "train.csv")) and os.path.exists(os.path.join(self.dataset_csv_path, "val.csv")):
            train_df = pd.read_csv(os.path.join(self.dataset_csv_path, "train.csv"))
            val_df = pd.read_csv(os.path.join(self.dataset_csv_path, "val.csv"))
        else:
            train_df = self.train_val_df.sample(frac=1-self.val_split_ratio)
            val_df = self.train_val_df.drop(train_df.index)
            train_df.to_csv(os.path.join(self.dataset_csv_path,"train.csv"), index=False)
            val_df.to_csv(os.path.join(self.dataset_csv_path,"val.csv"), index=False)
        train_data = TeethDataset(train_df, self.sample_num, transform=self.train_transform)
        val_data = TeethDataset(val_df, self.sample_num, transform=self.train_transform)
        test_data = TeethDataset(self.test_df, self.sample_num, transform=self.test_transform)
        return train_data, val_data, test_data

    def setup(self, stage: Optional[str] = None):
        hdf5_path = os.path.join(*self.data_set["0"][0].split("/")[:3], "h5/TeethDataset.hdf5")
        if os.path.exists(hdf5_path):
            train_data = H5TeethDataset("train", hdf5_path)
            val_data = H5TeethDataset("val", hdf5_path)
            test_data = H5TeethDataset("test", hdf5_path)
        else:
            train_data, val_data, test_data = self.enum_dataset()
        if stage == "fit" or stage is None:
            self.train_dataset = train_data
            self.val_dataset = val_data
        if stage == "test" or stage is None:
            self.test_dataset = test_data

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True, val=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, train=False, val=True)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: TeethDataset, train: bool = False, val: bool = False) -> DataLoader:
        return DataLoader(dataset,
                        batch_size=self.batch_size,
                        shuffle=True if train and val else False,
                        num_workers=self.num_workers,
                        pin_memory=True,
                        drop_last=True if train and val else False,
                        )
