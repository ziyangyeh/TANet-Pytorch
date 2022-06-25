from typing import Dict, List, Optional, Tuple, Callable
import os
import vedo
import trimesh
import torch
import numpy as np
from torch.utils.data import Dataset
from pytorch3d.transforms import *

from models.submodules import Tooth_Centering


class TeethDataset(Dataset):
    def __init__(self,
                data_root,
                transform: Optional[Callable] = None
                ):
        super(TeethDataset, self).__init__()
        self.data_root = data_root
        self.file_list = list(set(list(item.replace(item[-4:], "") for item in os.listdir(data_root))))
        self.transform = transform
        self.data = dict()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mesh = vedo.Mesh(os.path.join(self.data_root, self.file_list[idx]+".vtp"))
        self.data["target_X"], self.data["target_X_v"] = Tooth_Centering(mesh).get_pcds()
        self.data["X"], self.data["X_v"] = self.data["target_X"].clone(), self.data["target_X_v"].clone()
        self.data["6dof"] = torch.zeros(size=(self.data["X_v"].shape[0], 6))

        if self.transform is not None:
            self.data = self.transform(self.data)

        self.data["C"] = torch.cat([torch.from_numpy(trimesh.PointCloud(self.data["X_v"][idx, :, :].numpy()).centroid).unsqueeze(0) for idx in range(mesh.celldata["Label"].max())], dim=0).float()
        self.data["C"] = torch.cat((self.data["C"], torch.from_numpy(trimesh.PointCloud(self.data["X"][:(int)(self.data["X"].shape[0]/2), :].numpy()).centroid).unsqueeze(0), torch.from_numpy(trimesh.PointCloud(self.data["X"][(int)(self.data["X"].shape[0]/2):, :].numpy()).centroid).unsqueeze(0)), dim=0).float()

        for idx in range(self.data["X_v"].shape[0]):
            self.data["X_v"][idx] = self.data["X_v"][idx] - self.data["C"][idx]

        self.data["X"] = self.data["X_v"].reshape(self.data["target_X"].shape)

        return self.data
