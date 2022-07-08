from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn

from openpoints.models.build import MODELS, build_model_from_cfg

def _one_iter(A: torch.Tensor, embeddding: torch.Tensor, grus: nn.GRU)->torch.Tensor:
    h = [None] * A.shape[1]
    for i in range(A.shape[1]):
        m = torch.bmm(torch.cat((torch.cat((A[:, :i, :], A[:, i+1:, :]), dim=1)[:, :, :i], torch.cat((A[:, :i, :], A[:, i+1:, :]), dim=1)[:, :, i+1:]), dim=2), torch.cat((embeddding[:,:i],embeddding[:,i+1:]), dim=1))
        _, h_v = grus(m, embeddding[:, i, :].unsqueeze(0).repeat(grus.T_step, 1, 1))
        h[i] = h_v[-1].unsqueeze(1)
    return torch.cat(h, dim=1)

@MODELS.register_module()
class GRUs(nn.Module):
    def __init__(self,
                input_size,
                hidden_size,
                T_step,
                **kwargs
                ):
        super(GRUs, self).__init__()
        assert T_step is not None
        self.T_step=T_step
        self.grus = nn.GRU(input_size, hidden_size, T_step, batch_first=True)

    def forward(self, m: torch.Tensor, h_v: torch.Tensor)->Tuple[torch.Tensor, torch.Tensor]:
        out, h_v = self.grus(m, h_v)
        return out, h_v

@MODELS.register_module()
class FPM(nn.Module):
    def __init__(self,
                K_iter,
                rnn,
                **kwargs
                ):
        super(FPM, self).__init__()
        assert K_iter is not None
        self.K_iter=K_iter
        self.grus = build_grus(rnn, **kwargs)

    def forward(self, A: torch.Tensor, embeddding: torch.Tensor):
        current = embeddding
        for _ in range(self.K_iter):
            h = _one_iter(A, current, self.grus)
            current = h
        return h

def build_grus(cfg, **kwargs):
    return build_model_from_cfg(cfg, **kwargs)

def build_feature_propagation_module(cfg, **kwargs):
    return build_model_from_cfg(cfg, **kwargs)
