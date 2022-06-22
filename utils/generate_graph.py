import torch
import torch.nn as nn
import vedo
import numpy as np


class Learned_Matrix(nn.Module):
    def __init__(self,
                 upper_jaw: vedo.mesh.Mesh,
                 lower_jaw: vedo.mesh.Mesh,
                 device: torch.device
                 ):
        super(Learned_Matrix, self).__init__()
        self.upper_jaw = upper_jaw
        self.lower_jaw = lower_jaw
        self.device = device
        self.upper_centers, self.lower_centers = self._get_centers()

    def _get_centers(self):
        "trimesh/util.py -> len(index) => index.size"
        upper_centers=np.zeros((16,3))
        lower_centers=np.zeros((16,3))
        upper_centers[-2]=self.upper_jaw.centerOfMass()
        lower_centers[-2]=self.lower_jaw.centerOfMass()
        upper_centers[-1]=self.lower_jaw.centerOfMass()
        lower_centers[-1]=self.upper_jaw.centerOfMass()
        
        for i in range((self.upper_jaw.celldata["Label"]).max()):
            upper_centers[i]=self.upper_jaw.to_trimesh().submesh(np.where(self.upper_jaw.celldata["Label"]==i+1)[0], append=True).centroid
        
        for i in range((self.lower_jaw.celldata["Label"]).max()):
            lower_centers[i]=self.lower_jaw.to_trimesh().submesh(np.where(self.lower_jaw.celldata["Label"]==i+1)[0], append=True).centroid
        return torch.from_numpy(upper_centers).to(self.device), torch.from_numpy(lower_centers).to(self.device)

    def _get_learned_matrix(self):
        return torch.cdist(self.upper_centers, self.upper_centers), torch.cdist(self.lower_centers, self.lower_centers)
    
    def forward(self):
        return self._get_learned_matrix()
