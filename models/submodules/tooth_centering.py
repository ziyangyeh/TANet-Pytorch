from typing import Dict, List, Optional, Tuple, Callable
import vedo
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.utils.dlpack

class Tooth_Centering(nn.Module):
    def __init__(self,
                 mesh: vedo.Mesh,
                 sample_num=512,
                 ):
        super(Tooth_Centering, self).__init__()
        self.mesh = mesh
        self.sample_num = sample_num
        self.result = self.get_centers_and_pcds()

    
    def get_centers_and_pcds(self):
        "trimesh/util.py -> len(index) => index.size"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        C=np.zeros((15,3))
        C[-1]=self.mesh.centerOfMass()

        X = o3d.geometry.PointCloud()
        for i in range((self.mesh.celldata["Label"]).max()):
            tmp = self.mesh.to_trimesh().submesh(np.where(self.mesh.celldata["Label"]==i+1)[0], append=True)
            C[i]=tmp.centroid
            tooth = o3d.geometry.PointCloud()
            tooth.points = o3d.utility.Vector3dVector(np.asarray(tmp.as_open3d.sample_points_uniformly(number_of_points=self.sample_num).points)-C[i])
            X += tooth
        X_v = torch.from_numpy(np.asarray(X.points)).reshape(self.mesh.celldata["Label"].max(), self.sample_num, 3)
        if device == "cuda":
            X = torch.utils.dlpack.from_dlpack(o3d.t.geometry.PointCloud().from_legacy(X).cuda().point["positions"].to_dlpack())
        else:
            X = torch.utils.dlpack.from_dlpack(o3d.t.geometry.PointCloud().from_legacy(X).point["positions"].to_dlpack())
        return torch.from_numpy(C).to(device).type(torch.float32), X.type(torch.float32), X_v.to(device).type(torch.float32)

    def forward(self)->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.get_centers_and_pcds()
