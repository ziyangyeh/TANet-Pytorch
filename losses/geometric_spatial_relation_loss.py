import torch
import torch.nn as nn
from torch.autograd import Variable

from pytorch3d.transforms import *
from pytorch3d.loss import chamfer_distance
from mmdet3d.models.losses import ChamferDistance

class GeometricSpatialRelationLoss(nn.Module):
    def __init__(self, sigma: int, criterion_mode: str='l2'):
        super(GeometricSpatialRelationLoss, self).__init__()
        self.sigma = sigma
        self.criterion_mode = criterion_mode

    def forward(self, X_v, target_X_v, node_idx: int, device: torch.device):
        loss = torch.FloatTensor([0]).to(device)
        mid = (int)((node_idx-2)/2)
        c_X_v = torch.clamp(X_v, min=-self.sigma, max=self.sigma)
        c_target_X_v = torch.clamp(target_X_v, min=-self.sigma, max=self.sigma)
        for idx in range(node_idx):
            if idx < mid:
                sub_c_X_v = torch.cat([c_X_v[:idx, :, :], c_X_v[idx+1:, :, :], c_X_v[idx+mid, :, :].unsqueeze(0)],dim=0)
                sub_c_target_X_v = torch.cat([c_target_X_v[:idx, :, :], c_target_X_v[idx+1:, :, :], c_target_X_v[idx+mid, :, :].unsqueeze(0)],dim=0)
            elif mid <= idx < node_idx-2:
                sub_c_X_v = torch.cat([c_X_v[:idx, :, :], c_X_v[idx+1:, :, :], c_X_v[idx-mid, :, :].unsqueeze(0)],dim=0)
                sub_c_target_X_v = torch.cat([c_target_X_v[:idx, :, :], c_target_X_v[idx+1:, :, :], c_target_X_v[idx-mid, :, :].unsqueeze(0)],dim=0)
            elif idx == node_idx-2:
                sub_c_X_v = c_X_v[:mid, :, :]
                sub_c_target_X_v = c_target_X_v[:mid, :, :]
            elif idx == node_idx-1:
                sub_c_X_v = c_X_v[mid:, :, :]
                sub_c_target_X_v = c_target_X_v[mid:, :, :]
            sub_c_X_v = sub_c_X_v.view((sub_c_X_v.shape[0]*sub_c_X_v.shape[1], sub_c_X_v.shape[-1])).unsqueeze(0)
            sub_c_target_X_v = sub_c_target_X_v.view((sub_c_target_X_v.shape[0]*sub_c_target_X_v.shape[1], sub_c_target_X_v.shape[-1])).unsqueeze(0)
            if self.criterion_mode == 'l2':
                tmp,_ = chamfer_distance(sub_c_X_v, sub_c_target_X_v, batch_reduction="sum", point_reduction="sum")
            elif self.criterion_mode == 'smooth_l1':
                loss_src, loss_dst = ChamferDistance(reduction='sum', mode='smooth_l1').to(device)(sub_c_X_v, sub_c_target_X_v)
                tmp = loss_src+loss_dst
            else:
                raise NotImplementedError
            loss += tmp
        loss = Variable(loss, requires_grad=True)
        return loss
