import trimesh
import vedo
import open3d as o3d
import numpy as np

def to_origin_and_normalize(mesh: trimesh.base.Trimesh)->trimesh.base.Trimesh:
    mesh.vertices = mesh.vertices-mesh.centroid
    m = np.max(np.sqrt(np.sum(mesh.vertices ** 2, axis=1))) # Find the length of the major axis
    mesh.vertices = mesh.vertices / m
    return mesh

def global_rotation_matrix(mesh: trimesh.base.Trimesh, aug_num=50)->np.ndarray:
    return trimesh.transformations.random_rotation_matrix(num=aug_num)

def extract_teeth_without_gingiva(mesh: vedo.Mesh)->vedo.Mesh:
    mesh.deleteCellsByPointIndex(list(set(np.asarray(mesh.cells())[np.where(mesh.celldata["Label"]==0)[0]].flatten())))
    return mesh
