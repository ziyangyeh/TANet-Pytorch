from typing import Dict, List, Optional, Tuple, Callable
import vedo
import trimesh
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.utils.dlpack

class Tooth_Centering(nn.Module):
    def __init__(self,
                mesh: vedo.Mesh,
                sample_num=512,
                ):
        super(Tooth_Centering, self).__init__()
        self.mesh = mesh
        self.sample_num = sample_num
        self.max_label = self.mesh.celldata["Label"].max()
        self.total_points_num = self.sample_num*self.mesh.celldata["Label"].max()

    def get_max_num(self):
        return self.max_label
    
    # CPU Version
    def get_pcds(self):
        "trimesh/util.py -> len(index) => index.size"
        X = np.zeros((self.total_points_num,3), dtype=np.float32)
        for i in range((self.max_label)):
            tmp = self.mesh.to_trimesh().submesh(np.where(self.mesh.celldata["Label"]==i+1)[0], append=True)
            tooth = trimesh.sample.sample_surface_even(tmp, self.sample_num)[0]
            X[i*self.sample_num:(i*self.sample_num+self.sample_num)] = tooth
        X_v = X.reshape(self.max_label, self.sample_num, 3)

        return torch.from_numpy(X), torch.from_numpy(X_v)

    @torch.no_grad()
    def forward(self)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.get_pcds()
