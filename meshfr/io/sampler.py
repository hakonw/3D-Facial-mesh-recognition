import torch_geometric.data
import torch_geometric.utils
import torch
from scipy.spatial import Delaunay
from trimesh import Trimesh
import meshfr.datasets.reduction_transform as reduction_transform
import numpy as np

def data_all_sample(data):
    spatial = Delaunay(points=data[:, 0:2])
    faces = torch.tensor(spatial.simplices, dtype=torch.long).t().contiguous()
    pos = torch.from_numpy(data).to(torch.float).contiguous()
    d = torch_geometric.data.Data(pos=pos, face=faces)  # Note Face not Faces
    return d


def data_simple_sample(data, n_vertices=2048):
    # Sample n_vertices points
    n_rows = data.shape[0]
    random_indices = np.random.choice(n_rows, size=n_vertices, replace=False)
    reduced_data = data[random_indices, :]

    spatial = Delaunay(points=reduced_data[:, 0:2])
    
    faces = torch.tensor(spatial.simplices, dtype=torch.long).t().contiguous()
    pos = torch.from_numpy(reduced_data).to(torch.float).contiguous()
    d = torch_geometric.data.Data(pos=pos, face=faces)  # Note Face not Faces
    return d


def data_2pass_sample(data, n_vertices=2048, n_vertices_prefit=8192):
    assert n_vertices <= n_vertices_prefit

    # Sample n_vertices_prefit points
    n_rows = data.shape[0]
    random_indices = np.random.choice(n_rows, size=n_vertices_prefit, replace=False)
    reduced_data = data[random_indices, :]

    spatial = Delaunay(points=reduced_data[:, 0:2])

    tri = Trimesh(vertices=reduced_data, faces=spatial.simplices)
    tri = reduction_transform.simplify_trimesh(tri, n_vertices, 2)
    d = torch_geometric.utils.from_trimesh(tri)
    return d


def data_bruteforce_sample(data, n_vertices=2048):
    spatial = Delaunay(points=data[:, 0:2])  # Create mesh for everything

    tri = Trimesh(vertices=data, faces=spatial.simplices)
    tri = reduction_transform.simplify_trimesh(tri, n_vertices, 2)
    d = torch_geometric.utils.from_trimesh(tri)
    return d