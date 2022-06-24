import torch

def get_learned_matrix(centers: torch.Tensor):
        return torch.cdist(centers, centers)
