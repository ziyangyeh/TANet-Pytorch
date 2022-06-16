import __init__
import torch
import torch.nn as nn
from openpoints.utils import EasyConfig
from openpoints.models.build import build_model_from_cfg

cfg = EasyConfig()
cfg.load("./config/default.yaml", recursive=True)

model = build_model_from_cfg(cfg.model).to("cuda")

print(model.pose_regressor.mlps)


for name,param in model.pose_regressor.named_parameters():
    print(name, param)
    
x = torch.randn(size=(1,512)).to("cuda")
tmp = model.pose_regressor.mlps(x)
print(tmp)

print(model.Xi)