from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
from utils import get_learned_matrix
from models.submodules import build_jaw_encoder, build_tooth_encoder, build_pose_regressor, build_feature_propagation_module
from openpoints.models.build import MODELS


@MODELS.register_module()
class TANet(nn.Module):
    def __init__(self,
                 global_encoder_args=None,
                 local_encoder_args=None,
                 pose_regressor=None,
                 feature_propagation_module=None,
                 **kwargs):
        super().__init__()
        self.global_encoder = build_jaw_encoder(global_encoder_args, **kwargs)
        self.local_encoder = build_tooth_encoder(local_encoder_args, **kwargs)
        self.Xi = torch.nn.Parameter(torch.randn(pose_regressor.Xi_dim), requires_grad=False)
        self.pose_regressor = build_pose_regressor(pose_regressor, **kwargs)
        self.feature_propagation_module = build_feature_propagation_module(feature_propagation_module, **kwargs)

    def forward(self, X: Dict[str, torch.Tensor]):
        jaw_embedding = self.global_encoder(X["X"], X["C"])[1][-1].squeeze(-1)
        
        # TODO: 直接初始化一个Shape一样的Tensor
        teeth_embedddings = []
        for i in range(X["X_v"].shape[1]):
            tooth_embedding = self.local_encoder(X["X_v"][:, i, :, :].contiguous())
            teeth_embedddings.append(tooth_embedding[1][-1].unsqueeze(1))
        teeth_embedddings = torch.cat(teeth_embedddings, dim=1).squeeze(-1)
        print(teeth_embedddings.shape)
        teeth_embedddings = torch.cat((teeth_embedddings, torch.zeros(X["X_v"].shape[0], 2, X["X_v"].shape[2], device=teeth_embedddings.device)), dim= 1)
        return teeth_embedddings
        # learned_matrices = []
        # for i in range(teeth_embedddings.shape[0]):
        #     learned_matrices.append(get_learned_matrix(X["C"][i, :, :]).unsqueeze(0))
        # learned_matrices = torch.cat(learned_matrices, dim=0)
        
        # h = self.feature_propagation_module(learned_matrices, teeth_embedddings)
        
        # return self.pose_regressor(concated)
