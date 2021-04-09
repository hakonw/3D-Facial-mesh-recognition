
import torch_geometric.utils

class SimplifyQuadraticDecimationBruteForce(object):
    def __init__(self, vertices): 
        self.verticies = vertices 

    def __call__(self, data):
        trimesh = torch_geometric.utils.to_trimesh(data)
        for i in range(20):
                vertices = trimesh.vertices.shape[0]
                faces = trimesh.faces.shape[0]

                if vertices == self.verticies:
                    new_data = torch_geometric.utils.from_trimesh(trimesh)
                    data.face = new_data.face
                    data.pos = new_data.pos
                    return data  # Dont mess with any other properties
                if vertices < self.verticies:
                    raise AssertionError(f"Optimized too much. TODO fix {trimesh}")

                must_remove_vertices = vertices - self.verticies
                must_remove_faces = must_remove_vertices  # Possibly add a factor here
                trimesh = trimesh.simplify_quadratic_decimation(faces - must_remove_faces)