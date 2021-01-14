import torch
from torch_geometric.data import Data

def yield_file(in_file):
    f = open(in_file)
    buf = f.read()
    f.close()
    for b in buf.split('\n'):
        if b.startswith('v '):
            yield ['v', [float(x) for x in b.split(" ")[1:]]]
        elif b.startswith('f '):
            triangles = b.split(' ')[1:]
            if triangles[-1] == '':
                triangles = triangles[:-1]
            # -1 as .obj is base 1 but the Data class expects base 0 indices
            yield ['f', [int(t.split("/")[0]) - 1 for t in triangles]]
        else:
            yield ['', ""]


def read_obj(in_file, triangulation=True):
    vertices = []
    faces = []

    for k, v in yield_file(in_file):
        if k == 'v':
            vertices.append(v)
        elif k == 'f':
            faces.append(v)

    if not len(faces) or not len(vertices):
        return None

    if triangulation:
        assert len(faces[0]) >= 3
        faces_tri = []
        for face in faces:
            for i in range(1, len(face)-1): # For quad it is from 1 -> 4-1 (not included)
                tri_corner0 = face[0]
                tri_corner1 = face[i]
                tri_corner2 = face[i+1]
                faces_tri.append([tri_corner0, tri_corner1, tri_corner2])
        faces = faces_tri

    pos = torch.tensor(vertices, dtype=torch.float)
    face = torch.tensor(faces, dtype=torch.long).t().contiguous()

    data = Data(pos=pos, face=face)

    return data