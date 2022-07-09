import torch
import torch.nn as nn
from torch.autograd import Variable

from pytorch3d.transforms import *
from pytorch3d.loss import chamfer_distance
from mmdet3d.models.losses import ChamferDistance

class GeometricReconstructionLoss(nn.Module):
    def __init__(self, criterion_mode: str='l2'):
        super(GeometricReconstructionLoss, self).__init__()
        self.criterion_mode = criterion_mode

    def forward(self, X_v, target_X_v, rig_trans_matrix, device: torch.device):
        loss = torch.FloatTensor([0]).to(device)
        for idx in range(X_v.shape[0]):
            riged_tar = Transform3d(matrix=rig_trans_matrix[idx, :, :]).transform_points(target_X_v[idx, :, :]).unsqueeze(0)
            if self.criterion_mode == 'l2':
                tmp,_ = chamfer_distance(X_v[idx, :, :].unsqueeze(0), riged_tar, point_reduction="sum")
            elif self.criterion_mode == 'smooth_l1':
                loss_src, loss_dst = ChamferDistance(reduction='sum', mode='smooth_l1').to(device)(X_v[idx, :, :].unsqueeze(0), riged_tar)
                tmp = loss_src+loss_dst
            else:
                raise NotImplementedError
            loss += tmp
        loss = Variable(loss, requires_grad=True)
        return loss
