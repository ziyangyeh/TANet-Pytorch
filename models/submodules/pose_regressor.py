from typing import Dict, List, Optional, Tuple, Callable

import logging
import torch
import torch.nn as nn

from openpoints.models.build import MODELS, build_model_from_cfg

@MODELS.register_module()
class MLPs(nn.Module):
    def __init__(self,
                mlps,
                activators=None,
                dropouts=None,
                **kwargs
                ):
        super(MLPs, self).__init__()
        if mlps is not None:
            assert len(mlps)==len(activators)==len(dropouts)
            tmp = nn.ModuleList()
            tmp.append(nn.Flatten())
            for combs in zip(mlps, activators, dropouts):
                tmp.append(nn.Linear(combs[0][0], combs[0][1]))
                tmp.append(nn.ReLU() if combs[1] == "relu" else nn.Tanh())
                tmp.append(nn.Dropout(combs[2]))
        nn.init.zeros_(tmp[-1-2].weight.data)
        nn.init.zeros_(tmp[-1-2].bias.data)
        self.mlps = nn.Sequential(*tmp)

    def forward(self, x: torch.Tensor):
        x = self.mlps(x)
        return x

def build_pose_regressor(cfg, **kwargs):
    return build_model_from_cfg(cfg, **kwargs)
