
import torch_geometric.utils

def simplify_trimesh(trimesh, target, undershootmargin, overshootmargin=0):
    if trimesh.vertices.shape[0] <= target:
        return trimesh
    sizes = []
    for _ in range(20):
        vertices = trimesh.vertices.shape[0]
        faces = trimesh.faces.shape[0]

        sizes.append(vertices)
        if (target-undershootmargin) <= vertices <= (target+overshootmargin):
            return trimesh
        if vertices < target - undershootmargin:
            raise RuntimeError(f"Optimized too much. TODO fix {trimesh}, {sizes}")

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

from torch_geometric.transforms import Center
class NormalizeScale(object):
    r"""Centers and translates the  node positions to the interval :math:`(-1, 1)`.
    """

    def __init__(self):
        self.center = Center()

    def __call__(self, data):
        data = self.center(data)

        scale = (1 / data.pos.abs().max()) * 0.999999
        data.pos = data.pos * scale

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
    
import numbers
from itertools import repeat
import torch
class RandomTranslateScaled(object):
    r"""Translates node positions by randomly sampled translation values
    within a given interval. In contrast to other random transformations,
    translation is applied separately at each position.

    Args:
        translate (sequence or float or int): Maximum translation in each
            dimension, defining the range
            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
            If :obj:`translate` is a number instead of a sequence, the same
            range is used for each dimension.
    """

    def __init__(self, translate):
        self.translate = translate

    def __call__(self, data):
        (n, dim), t = data.pos.size(), self.translate
        scale = data.pos.abs().max() * 0.999999
        if isinstance(t, numbers.Number):
            t = t * scale
            t = list(repeat(t, times=dim))
        else: 
            t = [t0 * scale for t0 in t]
        assert len(t) == dim

        ts = []
        for d in range(dim):
            ts.append(data.pos.new_empty(n).uniform_(-abs(t[d]), abs(t[d])))

        data.pos = data.pos + torch.stack(ts, dim=-1)
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.translate)