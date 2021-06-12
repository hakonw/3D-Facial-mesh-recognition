
import torch_geometric.utils

def simplify_trimesh(trimesh, target, undershootmargin, overshootmargin=0):
    for _ in range(20):
        vertices = trimesh.vertices.shape[0]
        faces = trimesh.faces.shape[0]

        if (target-undershootmargin) <= vertices <= (target+overshootmargin):
            return trimesh
        if vertices < target - undershootmargin:
            raise RuntimeError(f"Optimized too much. TODO fix {trimesh}")

        must_remove_vertices = vertices - target
        must_remove_faces = must_remove_vertices # //2 -1  # Possibly add a factor here
        trimesh = trimesh.simplify_quadratic_decimation(faces - must_remove_faces)
    raise RuntimeError(
        f"Was not able to optimize within 20 generations. Stopping: {trimesh}")

class SimplifyQuadraticDecimationBruteForce(object):
    def __init__(self, vertices, undershootmargin=0):
        self.verticies = vertices 
        self.undershootmargin = undershootmargin

    def __call__(self, data):
        trimesh = torch_geometric.utils.to_trimesh(data)
        trimesh = simplify_trimesh(trimesh, self.verticiesm, self.undershootmargin)

        # Reconstruct and overwrite data
        # Dont mess with any other properties than pos and data
        new_data = torch_geometric.utils.from_trimesh(trimesh)
        data.pos = new_data.pos
        data.face = new_data.face
        return data

# Or use TORCH_GEOMETRIC.TRANSFORMS.DELAUNAY
import torch_geometric.data
import torch
from scipy.spatial import Delaunay
class DelaunayIt(object):
    def __call__(self, data):
        spatial = Delaunay(points=data.pos[:, 0:2], qhull_options='QJ')
        faces = torch.tensor(spatial.simplices, dtype=torch.long).t().contiguous()
        # pos = data.pos # torch.from_numpy(data).to(torch.float).contiguous()
        # d = torch_geometric.data.Data(pos=pos, face=faces)  # Note Face not Faces
        data.face = faces
        return data
    