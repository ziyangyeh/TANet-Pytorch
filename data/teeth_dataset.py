from typing import Dict, List, Optional, Tuple, Callable
import os
import vedo
import trimesh
import torch
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pytorch3d.transforms import *

from models.submodules import Tooth_Centering


class TeethDataset(Dataset):
    def __init__(self,
                df: pd.DataFrame,
                sample_num: int = 512,
                transform: Optional[Callable] = None
                ):
        super(TeethDataset, self).__init__()
        self.df = df
        self.sample_num = sample_num
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = dict()
        mesh = vedo.Mesh(self.df.values[idx][0])
        data["X"], data["X_v"], data["C"] = Tooth_Centering().get_pcds_and_centers(mesh, self.sample_num)
        data["target_X"], data["target_X_v"] = data["X"].clone(), data["X_v"].clone()
        data["6dof"] = torch.zeros(size=(data["X_v"].shape[0], 6))

        if self.transform is not None:
            data = self.transform(data)

        data["C"] = torch.cat([torch.from_numpy(trimesh.PointCloud(data["X_v"][i, :, :].numpy()).centroid).unsqueeze(0) for i in range(data["C"].shape[0])], dim=0)
        data["C"] = torch.cat((data["C"], torch.from_numpy(trimesh.PointCloud(data["X"][:(int)(data["X"].shape[0]/2), :].numpy()).centroid).unsqueeze(0), torch.from_numpy(trimesh.PointCloud(data["X"][(int)(data["X"].shape[0]/2):, :].numpy()).centroid).unsqueeze(0)), dim=0).float()

        return data

class H5TeethDataset(Dataset):
    def __init__(self,
                data_type: str,
                file_path: str,
                ):
        super(H5TeethDataset, self).__init__()
        self.data_type = data_type
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.length = len(f[data_type]["X"])

    def open_hdf5(self):
        self.hdf5 = h5py.File(self.file_path, 'r', libver='latest', swmr=True)
        self.dataset = self.hdf5[self.data_type]

    def __len__(self):
        return self.length
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not hasattr(self, 'hdf5'):
            self.open_hdf5()
        data = dict()
        data["X"] = torch.from_numpy(self.dataset["X"][idx])
        data["X_v"] = torch.from_numpy(self.dataset["X_v"][idx])
        data["target_X"] = torch.from_numpy(self.dataset["target_X"][idx])
        data["target_X_v"] = torch.from_numpy(self.dataset["target_X_v"][idx])
        data["C"] = torch.from_numpy(self.dataset["C"][idx])
        data["6dof"] = torch.from_numpy(self.dataset["6dof"][idx])

        return data
