from typing import Dict, List, Optional, Tuple, Callable
import os
import vedo
import trimesh
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pytorch3d.transforms import *

from models.submodules import Tooth_Centering


class TeethDataset(Dataset):
    def __init__(self,
                df: pd.DataFrame,
                transform: Optional[Callable] = None
                ):
        super(TeethDataset, self).__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = dict()
        mesh = vedo.Mesh(self.df.values[idx][0])
        data["X"], data["X_v"], data["C"] = Tooth_Centering(mesh).get_pcds_and_centers()
        data["target_X"], data["target_X_v"] = data["X"].clone(), data["X_v"].clone()
        data["6dof"] = torch.zeros(size=(data["X_v"].shape[0], 6))

        if self.transform is not None:
            data = self.transform(data)

        data["C"] = torch.cat((data["C"], torch.from_numpy(trimesh.PointCloud(data["X"][:(int)(data["X"].shape[0]/2), :].numpy()).centroid).unsqueeze(0), torch.from_numpy(trimesh.PointCloud(data["X"][(int)(data["X"].shape[0]/2):, :].numpy()).centroid).unsqueeze(0)), dim=0).float()

        return data
