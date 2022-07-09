from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from pytorch3d.transforms import *

class Tooth_Assembler(nn.Module):
    def __init__(self):
        super(Tooth_Assembler, self).__init__()

    def forward(self, X: Dict[str, torch.Tensor], dofs: torch.Tensor, device: torch.device) -> torch.Tensor:
        assembled = torch.zeros(size=X["X_v"].shape, device=device)
        pred_matrices = torch.cat([se3_exp_map(dofs[idx]).unsqueeze(0) for idx in range(dofs.shape[0])], dim=0)
        gt_matrices = torch.cat([se3_exp_map(X["6dof"][idx]).unsqueeze(0) for idx in range(X["6dof"].shape[0])], dim=0)
        pred2gt_matrices = torch.zeros(size=(gt_matrices.shape), device=device)

        for idx in range(X["X_v"].shape[1]): # X_v: 8,28,512,3; matrices: B,28,4,4
            assembled[:, idx, :, :] = Transform3d(matrix=pred_matrices[:, idx, :, :]).to(device).transform_points(X["X_v"][:, idx, :, :])
            pred2gt_matrices[:, idx, :, :] = Transform3d.compose(Transform3d(matrix=Transform3d(matrix=pred_matrices[:, idx, :, :]).inverse().get_matrix()),
                                                            Transform3d(matrix=gt_matrices[:, idx, :, :])).inverse().get_matrix()
        return assembled, pred2gt_matrices
