from typing import Dict, List, Optional, Tuple, Callable
import vedo
import trimesh
import numpy as np
import torch
import torch.nn as nn
import torch.utils.dlpack

class Tooth_Centering(nn.Module):
    def __init__(self):
        super(Tooth_Centering, self).__init__()

    # CPU Version
    def get_pcds_and_centers(self, mesh: vedo.Mesh, sample_num=512):
        "trimesh/util.py -> len(index) => index.size"
        max_label = mesh.celldata["Label"].max()
        total_points_num = sample_num*mesh.celldata["Label"].max()
        C = np.zeros((max_label, 3), dtype=np.float32)
        X = np.zeros((total_points_num, 3), dtype=np.float32)
        for i in range(max_label):
            tmp = mesh.to_trimesh().submesh(np.where(mesh.celldata["Label"]==i+1)[0], append=True)
            tooth = trimesh.sample.sample_surface_even(tmp, sample_num)[0]
            C[i] = trimesh.points.PointCloud(vertices=tooth).centroid
            X[i*sample_num:(i*sample_num+sample_num)] = tooth
        X_v = X.reshape(max_label, sample_num, 3)

        return torch.from_numpy(X), torch.from_numpy(X_v), torch.from_numpy(C)

    @torch.no_grad()
    def forward(self, X: Dict[str, torch.Tensor], device: torch.device)->Dict[str, torch.Tensor]:
        C = dict()
        C["X_v"] = torch.zeros(X["X_v"].shape, dtype=torch.float32, device=device)
        for batch_idx in range(X["X_v"].shape[0]):
            for tooth_idx in range(X["X_v"].shape[1]):
                C["X_v"][batch_idx][tooth_idx] = X["X_v"][batch_idx][tooth_idx]-X["C"][batch_idx][tooth_idx]
        C["X"] = C["X_v"].clone().reshape(X["X"].shape)
        C["C"] = X["C"]
        C["6dof"] = X["6dof"]
        return C
