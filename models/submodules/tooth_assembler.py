from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from pytorch3d.transforms import *

class Tooth_Assembler(nn.Module):
    def __init__(self):
        super(Tooth_Assembler, self).__init__()

    @torch.no_grad()
    def forward(self, X: Dict[str, torch.Tensor], dofs: torch.Tensor, device: torch.device) -> torch.Tensor:
        assembled = torch.zeros(size=X["X_v"].shape, device=device)
        #matrices = torch.cat([se3_exp_map(dofs[idx]).unsqueeze(0) for idx in range(dofs.shape[0])], dim=0)
        matrices = torch.cat([se3_exp_map(X["6dof"][idx]).unsqueeze(0) for idx in range(X["6dof"].shape[0])], dim=0)

        R = matrices[:, :, :3, :3] # 8,28,3,3
        T = matrices[:, :, 3, :3] # 8,28,3
        for idx in range(X["X_v"].shape[1]): # X_v: 8,28,512,3; matrices: 8,28,4,4
            assembled[:, idx, :, :] = Transform3d().compose(Rotate(R[:, idx, :, :]),
                                                            Translate(T[:, idx, :]),
                                                            ).to(device).transform_points(assembled[X["X_v"][:, idx, :, :]])
        return assembled
