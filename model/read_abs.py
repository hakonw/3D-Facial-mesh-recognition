import numpy as np


def read_abs_raw(filename):
    f = open(filename, "r")
    
    # The matlab code is terrible imo, and this is more restrictive
    line = f.readline().rstrip().split()
    rows = int(line[0])
    # print("rows", rows)

    line = f.readline().rstrip().split()
    colums = int(line[0])
    # print("columns", colums)

    # Throw away trash line
    f.readline()

    # FL: is the flags vector specifying if a point is valid
    flags = np.loadtxt(f, dtype=np.uint16, max_rows=1)
    assert(flags.shape[0] == rows*colums)
    # print("flags:", flags.shape)

    # X,Y,Z are matrices representing the 3D co-ords of each point
    data = np.loadtxt(f, dtype=np.float64, max_rows=3)
    assert(data.shape[0] == 3)
    assert(data.shape[1] == rows*colums)
    # print("data", data.shape)

    f.close()  # Close file as it now has been read

    # Remove invalid data and transpose to [Npoints, Ndim]
    data = data[:, flags == 1].transpose()
    # print(data.shape)

    return data

if __name__ == "__main__":
    data = read_abs_raw("02463d452.abs")
    
    # See how hybrid sampling changes it
    from scipy.spatial import Delaunay
    from trimesh import Trimesh
    import reduction_transform
    # Sample n_vertices_prefit points
    n_rows = data.shape[0]
    random_indices = np.random.choice(
        n_rows, size=3072, replace=False)
    reduced_data = data[random_indices, :]

    spatial = Delaunay(points=reduced_data[:, 0:2])

    tri = Trimesh(vertices=reduced_data, faces=spatial.simplices)
    tri = reduction_transform.simplify_trimesh(tri, 2048, 2)
    tri.export("02463d452-reduced.ply")
