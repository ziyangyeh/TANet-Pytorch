import torch
import torch.nn as nn
from models.submodules.jaw_encoder import build_jaw_encoder
from models.submodules.tooth_encoder import build_tooth_encoder
from models.submodules.pose_regressor import build_pose_regressor
from openpoints.models.build import MODELS


@MODELS.register_module()
class TANet(nn.Module):
    def __init__(self,
                 global_encoder_args=None,
                 local_encoder_args=None,
                 pose_regressor=None,
                 **kwargs):
        super().__init__()
        self.global_encoder = build_jaw_encoder(global_encoder_args, **kwargs)
        self.local_encoder = build_tooth_encoder(local_encoder_args, **kwargs)
        self.Xi = torch.nn.Parameter(torch.randn(pose_regressor.Xi_dim), requires_grad=False)
        self.pose_regressor = build_pose_regressor(pose_regressor, **kwargs)

    def forward(self,X):
        return 0
