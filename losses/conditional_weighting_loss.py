import torch
import torch.nn as nn
from torch.autograd import Variable

from .geometric_reconstruction_loss import GeometricReconstructionLoss
from .geometric_spatial_relation_loss import GeometricSpatialRelationLoss


class ConditionalWeightingLoss(nn.Module):
    def __init__(self, sigma: int=5, criterion_mode: str='l2'):
        super(ConditionalWeightingLoss, self).__init__()
        self.criterion_mode = criterion_mode
        self.recon_loss = GeometricReconstructionLoss(self.criterion_mode)
        self.spatial_loss = GeometricSpatialRelationLoss(sigma=sigma, criterion_mode=self.criterion_mode)

    def forward(self, X_v, target_X_v, rig_trans_matrix, node_idx: int, Xi, device: torch.device):
        loss = torch.FloatTensor(torch.zeros((X_v.shape[0],))).to(device)
        for idx in range(X_v.shape[0]):
            loss_recon = self.recon_loss(X_v[idx], target_X_v[idx], rig_trans_matrix[idx], device)
            loss_spatial = self.spatial_loss(X_v[idx], target_X_v[idx], node_idx, device)
            tmp = torch.min((loss_recon+loss_spatial)*(1/torch.exp(torch.norm(Xi[idx], dim=1))))
            loss[idx] = tmp
        loss = Variable(torch.sum(loss)/X_v.shape[0], requires_grad=True)
        return loss
