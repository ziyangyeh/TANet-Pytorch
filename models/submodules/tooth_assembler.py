from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from pytorch3d.transforms import *

class Tooth_Assembler(nn.Module):
    def __init__(self):
        super(Tooth_Assembler, self).__init__()

    def forward(self, X: Dict[str, torch.Tensor], dofs: torch.Tensor, device: torch.device) -> torch.Tensor:
        assembled = torch.zeros(size=X["X_v"].shape, device=device)
        matrices = torch.cat([se3_exp_map(dofs[idx]).unsqueeze(0) for idx in range(dofs.shape[0])], dim=0)
        # matrices = torch.cat([se3_exp_map(X["6dof"][idx]).unsqueeze(0) for idx in range(X["6dof"].shape[0])], dim=0)

        for idx in range(X["X_v"].shape[1]): # X_v: 8,28,512,3; matrices: B,28,4,4
            assembled[:, idx, :, :] = Transform3d(matrix=matrices[:, idx, :, :]).to(device).transform_points(X["X_v"][:, idx, :, :])

        return assembled
