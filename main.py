import open3d as o3d
import torch
import torch.nn as nn
from pytorch3d.transforms import *
from pytorch3d.ops import ball_query

from data import TeethDataset

# from openpoints.utils import EasyConfig
# from openpoints.models.build import build_model_from_cfg
# cfg = EasyConfig()
# cfg.load("config/default.yaml", recursive=True)

# model = build_model_from_cfg(cfg.model)
# model.cuda()

# teeth_dataset = LitDataModule("tmp/teeth_seg/augmented", 2, 1).train_dataloader()

# loss = nn.SmoothL1Loss(reduction="sum")
# for idx, item in enumerate(teeth_dataset):
#     print(item["X"].shape)
#     item = {k: v.to(device='cuda', non_blocking=True) for k, v in item.items()}
#     print(model(item).shape)
#     break
    # print(item["C"].shape)
    # print(item["X"].shape)
    # print(item[0]["X"])
    # print(ball_query(item[0]["X"], item[0]["X"], K=1)[2].shape)
    # print(ball_query(item[0]["X"], item[0]["X"], K=1)[2].squeeze(2))
    # print(ball_query(item[0]["X"], item[0]["X"], K=1)[0].shape)
    # print(ball_query(item[0]["X"], item[0]["X"], K=1)[0].squeeze(2))
    # print(ball_query(item[0]["X"], item[0]["X"], K=1)[0].shape)
    # print(ball_query(item[0]["X"], item[0]["X"], K=1)[0].squeeze(2))
    # print(loss(item[0]["X"], ball_query(item[0]["X"], item[0]["X"], K=1, radius=0.05)[2].squeeze(2)))
    # print(loss(item[0]["X_v"][0], ball_query(item[0]["X_v"][0], item[0]["X_v"][0], K=1, radius=0.2)[2].squeeze(2)))
    # print(item[0]["X_v"].shape)
    # print(item[0]["6dof"].shape)
    # print(item[0]["6dof"])
    # print(item[0]["X"][:, :512, :].shape)
    # tmp = Transform3d().compose(Translate(torch.randn(1,3)),
    #                             RotateAxisAngle(angle=torch.randint(-30, 30, (1,)), axis="X"),
    #                             RotateAxisAngle(angle=torch.randint(-30, 30, (1,)), axis="Y"),
    #                             RotateAxisAngle(angle=torch.randint(-30, 30, (1,)), axis="Z"),
    #                             ).get_matrix()
    # print(se3_log_map(tmp))

    # print(item[0]["target_X_v"][0][0].shape)
    # print(item[0]["X_v"][0][0].shape)
    # # for i in range(14):
    # #     target_X = o3d.geometry.PointCloud()
    # #     target_X.points = o3d.utility.Vector3dVector(item[0]["target_X_v"][0][i].cpu().detach().numpy())
    # #     o3d.io.write_point_cloud(f"./tmp/test_data/T_X_{i}.ply", target_X)
        
    # #     X_v = o3d.geometry.PointCloud()
    # #     X_v.points = o3d.utility.Vector3dVector(item[0]["X_v"][0][i].cpu().detach().numpy())
    # #     o3d.io.write_point_cloud(f"./tmp/test_data/X_{i}.ply", X_v)
    # break



# from openpoints.utils import EasyConfig
# from openpoints.models.build import build_model_from_cfg
# cfg = EasyConfig()
# cfg.load("config/default.yaml", recursive=True)

# model = build_model_from_cfg(cfg.model)
# model.cuda()

# batch_size = 20
# X=dict()
# X["C"] = torch.randn(batch_size, 30, 3).cuda()
# X["X"] = torch.randn(batch_size, 14336, 3).cuda()
# X["X_v"] = torch.randn(batch_size, 28, 512, 3).cuda()

# tmp = model(X)
# print(tmp)
# print(tmp.shape)

# tmp_1, tmp_2 = model(X)
# print(tmp_1)
# print(tmp_1.shape)
# print(tmp_2)
# print(tmp_2.shape)
# tmp = Transform3d().compose(Translate(torch.randn(1,3)),
#                             RotateAxisAngle(angle=torch.randint(-30, 30, (1,)), axis="X"),
#                             RotateAxisAngle(angle=torch.randint(-30, 30, (1,)), axis="Y"),
#                             RotateAxisAngle(angle=torch.randint(-30, 30, (1,)), axis="Z"),
#                             )

# print(tmp.transform_points(X["X"]).shape)
