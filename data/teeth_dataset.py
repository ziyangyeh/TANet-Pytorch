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

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = dict()
        mesh = vedo.Mesh(os.path.join(self.data_root, self.file_list[idx]+".vtp"))
        data["target_X"], data["target_X_v"] = Tooth_Centering(mesh).get_pcds()
        data["X"], data["X_v"] = data["target_X"].clone(), data["target_X_v"].clone()
        data["6dof"] = torch.zeros(size=(data["X_v"].shape[0], 6))

        if self.transform is not None:
            data = self.transform(data)

        data["C"] = torch.cat([torch.from_numpy(trimesh.PointCloud(data["X_v"][idx, :, :].numpy()).centroid).unsqueeze(0) for idx in range(mesh.celldata["Label"].max())], dim=0).float()
        data["C"] = torch.cat((data["C"], torch.from_numpy(trimesh.PointCloud(data["X"][:(int)(data["X"].shape[0]/2), :].numpy()).centroid).unsqueeze(0), torch.from_numpy(trimesh.PointCloud(data["X"][(int)(data["X"].shape[0]/2):, :].numpy()).centroid).unsqueeze(0)), dim=0).float()

        for idx in range(data["X_v"].shape[0]):
            data["X_v"][idx] = data["X_v"][idx] - data["C"][idx]

        return data
