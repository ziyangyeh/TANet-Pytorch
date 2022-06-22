from typing import List, Optional

import logging
import torch
import torch.nn as nn

from openpoints.models.build import MODELS, build_model_from_cfg

@MODELS.register_module()
class GRUs(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 T_step,
                 K_iter,
                 **kwargs
                 ):
        super(GRUs, self).__init__()
        assert T_step is not None or K_iter is not None
        self.T_step=T_step
        self.K_iter=K_iter
        self.grus = nn.GRU(input_size, hidden_size, T_step, batch_first=True)

    def forward(self, m, h_v):
        _, h_v = self.grus(m, h_v)
        return h_v

def build_grus(cfg, **kwargs):
    return build_model_from_cfg(cfg, **kwargs)

def build_feature_propagation_module(cfg, **kwargs):
    return build_model_from_cfg(cfg, **kwargs)