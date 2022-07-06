import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch3d.loss import *

class GeometricReconstructionLoss(nn.Module):
    def __init__(self):
        super(GeometricReconstructionLoss, self).__init__()

    def forward(self, X_v, target_X_v):
        loss,_ = chamfer_distance(X_v[:, 0, :, :], target_X_v[:, 0, :, :], batch_reduction='mean', point_reduction='sum')
        for idx in range(X_v.shape[0]):
            tmp,_ = chamfer_distance(X_v[:, idx, :, :], target_X_v[:, idx, :, :], batch_reduction='mean', point_reduction='sum')
            loss += tmp
        loss = Variable(loss, requires_grad=True)
        return loss
