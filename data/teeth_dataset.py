from typing import Dict, List, Optional, Tuple, Callable
import os
import vedo
import torch
from torch.utils.data import Dataset
import numpy as np

from models.submodules import Tooth_Centering
from utils.rearrange import rearrange


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
        mesh = vedo.Mesh(os.path.join(self.data_root, self.file_list[idx]+".stl"))
        mesh.celldata["Label"] = rearrange(np.loadtxt(os.path.join(self.data_root, self.file_list[idx]+".txt")).astype(int))
        mesh.deleteCellsByPointIndex(set(np.asarray(mesh.cells())[mesh.celldata["Label"]==0].flatten()))
        self.data["C"], self.data["X"], self.data["X_v"] = Tooth_Centering(mesh).forward()
        if self.transform is not None:
            self.data = self.transform(self.data)
        return self.data
