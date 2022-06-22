import __init__
import torch
import torch.nn as nn
from openpoints.utils import EasyConfig
from openpoints.models.build import build_model_from_cfg

cfg = EasyConfig()
cfg.load("./config/default.yaml", recursive=True)

model = build_model_from_cfg(cfg.model).to("cuda")

# print(model.global_encoder)
# print(model.local_encoder)
# print(model.pose_regressor)
# print(model.grus)


# for name,param in model.pose_regressor.named_parameters():
#     print(name, param)

# x = torch.randn(size=(1,512)).to("cuda")
# tmp = model.pose_regressor(x)
# print(tmp)

# print(model.Xi)

for name,param in model.grus.named_parameters():
    print(name, param)

# A = torch.randn(size=(15,15)).to("cuda")
# h_n = torch.randn(size=(15,512)).to("cuda")
# m = torch.mm(A, h_n)
# m = m.unsqueeze(0)
# print(m.shape)
# # m = torch.randn(size=(1,1,512)).to("cuda")
# h_v = torch.randn(size=(3,1,512)).to("cuda")
# tmp_h = model.grus(m, h_v)
# print(tmp_h[-1,:,:])
# print(model.K)