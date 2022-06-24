import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from data import TeethDataset
from openpoints.utils import EasyConfig
from openpoints.models.build import build_model_from_cfg

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = EasyConfig()
cfg.load("./config/default.yaml", recursive=True)

model = build_model_from_cfg(cfg.model).to(device)

# print(model)
# print(model.global_encoder)
# print(model.local_encoder)
# print(model.pose_regressor)
# print(model.feature_propagation_module)

# batch_size = 5

# jaw = torch.randn(size=(batch_size, 14336, 3)).contiguous().to(device)
# C = torch.randn(size=(batch_size, 14 , 3)).contiguous().to(device)
# jaw_embedding = model.global_encoder(jaw, C)
# print("global_encoder: " + str(jaw_embedding[1][-1].squeeze(-1).shape))

# teeth = torch.randn(size=(batch_size, 14, 512, 3)).to(device)
# teeth_embedddings = []
# for i in range(teeth.shape[1]):
#     tooth_embedding = model.local_encoder(teeth[:, i, :, :].contiguous())
#     teeth_embedddings.append(tooth_embedding[1][-1].unsqueeze(1))
# teeth_embedddings = torch.cat(teeth_embedddings, dim=1).squeeze(-1)
# teeth_embedddings = torch.cat((teeth_embedddings, torch.zeros(batch_size, 2, 512).to(device)), dim= 1)
# print("local_encoder: " + str(teeth_embedddings.shape))

# x = torch.randn(size=(batch_size,1024+512)).to(device)
# tmp = model.pose_regressor(x)
# print("pose_regressor: " + str(tmp.shape))

# learned_matrices = []
# centers = torch.randn(size=(batch_size, 16, 3)).to(device)
# for i in range(teeth_embedddings.shape[0]):
#      learned_matrices.append(torch.cdist(centers[i, :, :], centers[i, :, :]).unsqueeze(0))
# learned_matrices = torch.cat(learned_matrices, dim=0)
# print("learned_matrices: " + str(learned_matrices.shape))


# m = torch.cat(m, dim=0)

# h = model.feature_propagation_module(learned_matrices, teeth_embedddings)
# print("h: ", str(h.shape))

# h_v = torch.randn(size=(3, batch_size, 512)).to(device)
# tmp_o, tmp_h_v = model.feature_propagation_module.grus(m, h_v)
# print("gru_output: " + str(tmp_o.shape))
# print("gru_hidden: " + str(tmp_h_v.shape))
# print(model.K)

# del_row_col = torch.cat((teeth_embedddings[:, :i, :], teeth_embedddings[:, i+1:, :]), dim=1)
#     del_row_col = torch.cat((teeth_embedddings[:, :, :i], teeth_embedddings[:, :, i+1:]), dim=2)
#     m.append(del_row_col)
#     print(del_row_col.shape)

data_root = "./tmp/teeth_seg"
dataset = TeethDataset(data_root)
dataloader = DataLoader(dataset, batch_size=1)

for idx, item in enumerate(dataloader):
    print(item["X_v"].device)
    output = model(item)
    print(output.shape)
    break
