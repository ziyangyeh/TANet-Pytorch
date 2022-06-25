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
        super(TANet, self).__init__()
        self.global_encoder = build_jaw_encoder(global_encoder_args, **kwargs)
        self.local_encoder = build_tooth_encoder(local_encoder_args, **kwargs)
        self.feature_propagation_module = build_feature_propagation_module(feature_propagation_module, **kwargs)
        self.Xi_dim = pose_regressor.Xi_dim
        self.pose_regressor = build_pose_regressor(pose_regressor, **kwargs)

    def forward(self, X: Dict[str, torch.Tensor])->torch.Tensor:
        jaw_embedding = self.global_encoder(X["X"].contiguous(), X["C"][:, :-2, :].contiguous())[1][-1].squeeze(-1)

        teeth_embedddings = [None] * X["X_v"].shape[1]
        for i in range(X["X_v"].shape[1]):
            teeth_embedddings[i] = self.local_encoder(X["X_v"][:, i, :, :].contiguous())[1][-1].unsqueeze(1).squeeze(-1)
        teeth_embedddings = torch.cat(teeth_embedddings, dim=1)
        teeth_embedddings = torch.cat((teeth_embedddings, torch.zeros(X["X_v"].shape[0], 2, X["X_v"].shape[2], device=teeth_embedddings.device)), dim=1)

        upper_c = torch.cat((X["C"][:, :(int)((teeth_embedddings.shape[1]-2)/2), :], X["C"][:, -2, :].unsqueeze(1), X["C"][:, -1, :].unsqueeze(1)), dim=1)
        lower_c = torch.cat((X["C"][:, (int)((teeth_embedddings.shape[1]-2)/2):-2, :], X["C"][:, -1, :].unsqueeze(1), X["C"][:, -2, :].unsqueeze(1)), dim=1)

        upper_learned_matrices = [None] * upper_c.shape[0]
        lower_learned_matrices = [None] * lower_c.shape[0]
        for i in range(teeth_embedddings.shape[0]):
            upper_learned_matrices[i] = get_learned_matrix(upper_c[i]).unsqueeze(0)
            lower_learned_matrices[i] = get_learned_matrix(lower_c[i]).unsqueeze(0)
        upper_learned_matrices = torch.cat(upper_learned_matrices, dim=0)
        lower_learned_matrices = torch.cat(lower_learned_matrices, dim=0)
        upper_learned_matrices[:, :, -1][:,:-2] = 0
        lower_learned_matrices[:, :, -1][:,:-2] = 0
        upper_learned_matrices[:, -1, :][:,:-2] = 0
        lower_learned_matrices[:, -1, :][:,:-2] = 0

        upper_teeth_embedddings = torch.cat((teeth_embedddings[:, :(int)((teeth_embedddings.shape[1]-2)/2)], teeth_embedddings[:, -2].unsqueeze(1), teeth_embedddings[:, -1].unsqueeze(1)), dim=1)
        lower_teeth_embedddings = torch.cat((teeth_embedddings[:, (int)((teeth_embedddings.shape[1]-2)/2):-2], teeth_embedddings[:, -1].unsqueeze(1), teeth_embedddings[:, -2].unsqueeze(1)), dim=1)

        h_u = self.feature_propagation_module(upper_learned_matrices, upper_teeth_embedddings)
        upper_teeth_embedddings = upper_teeth_embedddings + h_u
        h_l = self.feature_propagation_module(lower_learned_matrices, lower_teeth_embedddings)
        lower_teeth_embedddings = lower_teeth_embedddings + h_l

        Xi = torch.randn(1, self.Xi_dim, device = teeth_embedddings.device).repeat(teeth_embedddings.shape[0], teeth_embedddings.shape[1]-2, 1)
        jaw_embedding = jaw_embedding.unsqueeze(1).repeat(1, teeth_embedddings.shape[1]-2, 1)
        concated = torch.cat([Xi, jaw_embedding, teeth_embedddings[:, :-2, :]], dim=2)

        dof = [None] * concated.shape[1]
        for i in range (concated.shape[1]):
            dof[i] = self.pose_regressor(concated[:, i, :]).unsqueeze(1)
        dof = torch.cat(dof, dim=1)

        return dof
