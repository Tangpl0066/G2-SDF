import torch
import numpy as np
from skimage import measure
import trimesh

Data = torch.load('./results/256_best_sdf1.pt')
voxel_matrix = np.array([item.cpu().detach().numpy() for item in Data])

verts, faces, normals, values = measure.marching_cubes(voxel_matrix, level=0, spacing=(1,1,1))
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.export('./results/256_best_sdf1.obj')
