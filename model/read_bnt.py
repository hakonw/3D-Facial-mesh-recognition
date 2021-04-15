import numpy as np
import torch_geometric.data
import torch_geometric.utils
import torch
from scipy.spatial import Delaunay


def read_bnt_raw(filename):
    f = open(filename, "r")

    nrows = np.fromfile(f, dtype=np.uint16, count=1)[0]
    ncols = np.fromfile(f, dtype=np.uint16, count=1)[0]
    zmin = np.fromfile(f, dtype=np.float64, count=1)[0]
    # print("nrows:", nrows, ", ncols:", ncols, ", zmin:", zmin)

    imfile_length = np.fromfile(f, dtype=np.uint16, count=1)[0]
    imfile_name = np.fromfile(f, dtype=np.uint8, count=imfile_length).tobytes().decode("ascii")
    # print("imfile_length", imfile_length, ", imfile_name", imfile_name)

    # (STOLEN from doc)
    # Normally, size of data must be nrows*ncols*5
    # data: Nx5 matrix where columns are 3D coordinates and 2D
    #   normalized image coordinates respectively. 2D coordinates are
    #   normalized to the range[0, 1]. N = nrows*ncols. In this matrix, values
    #   that are equal to zmin denotes the background.

    data_len = np.fromfile(f, dtype=np.uint32, count=1)[0]
    # print("data_len", data_len)
    assert((data_len/5).is_integer()) # To check that it didnt leave any data behind

    # note (5, data_len//5) instead of what matlab does. Because thats the pythonic way
    data = np.fromfile(f, dtype=np.float64, count=data_len).reshape((5, data_len//5))
    data = np.transpose(data)  # Make it more resonable
    # Check doc statement
    assert data.shape[0] * data.shape[1] == data_len
    # assert data.shape[0] == nrows*ncols  # "Normally". It dont do that often
    f.close()  # Close file as it now has been read

    # Trash the 2D data. No need for that
    data = data[:, 0:3]
    # print(data.shape)

    # Remove invalid points
    data = data[~np.all(data == zmin, axis=1)]
    data = data[~np.any(data == zmin, axis=1)]  # Where some files with invalid data...
    # assert ~np.any(data == zmin)  # Assert (all dims) for semi-invalid points
    return data


def data_simple_sample(data, n_vertices=2048):
    # Sample n_vertices points
    n_rows = data.shape[0]
    random_indices = np.random.choice(n_rows, size=n_vertices, replace=False)
    reduced_data = data[random_indices, :]

    spatial = Delaunay(points=reduced_data[:, 0:2])
    
    faces = torch.tensor(spatial.simplices, dtype=torch.long).t().contiguous()
    pos = torch.from_numpy(reduced_data).to(torch.float).contiguous()
    d = torch_geometric.data.Data(pos=pos, faces=faces)
    return d


from trimesh import Trimesh
import reduction_transform
def data_2pass_sample(data, n_vertices=2048, n_vertices_prefit=3072):
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


if __name__ == "__main__":
    filename = "./bs104_N_N_3.bnt"

    from datetime import datetime

    start = datetime.now()
    for _ in range(10):
        data = read_bnt_raw(filename)
        data_simple_sample(data)
    stop = datetime.now()
    print(f"Time simple: {stop-start}")
    # Time simple with shuffle: 0:00:00.411078
    # Time simple with choice:  0:00:00.121709
    # Time simple data:         0:00:00.070138

    start = datetime.now()
    for _ in range(10):
        data = read_bnt_raw(filename)
        data_2pass_sample(data)
    stop = datetime.now()
    print(f"Time hybrid: {stop-start}")
    # Time hybrid with shuffle: 0:00:02.271874
    # Time hybrid with choice : 0:00:01.951697
    # Time hybrid data:         0:00:01.992233

    data = read_bnt_raw(filename)
    start = datetime.now()
    for _ in range(10):
        np.random.shuffle(data)
    stop = datetime.now()
    print(f"Time shuffle: {stop-start}")
    # Time shuffle: 0:00:00.285436

    data = read_bnt_raw(filename)
    start = datetime.now()
    for _ in range(10):
        n_rows = data.shape[0]
        random_indices = np.random.choice(n_rows, size=2048, replace=False)
        reduced_data = data[random_indices, :]
    stop = datetime.now()
    print(f"Time choice: {stop-start}")
    # Time choice: 0:00:00.005260
