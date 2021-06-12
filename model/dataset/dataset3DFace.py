import os.path
from glob import glob
import pickle
from tqdm import tqdm
import torch_geometric.utils
import torch_geometric.io
import reduction_transform


def generate_3dface_dict(root, sample="2pass", sample_size=2048):
    dataset = {}
    
    # There are no sub folders, as for every date, every ident is in one folder
    # Find all sub-folders
    # Which are all the identities

    # Find all sub-folders
    # Which are all the identities
    folders = []
    for root, dirs, filenames in os.walk(root):
        folders = sorted(dirs)
        break  # prevent descending into subfolders

    # Find all scans for each identity
    pbar = tqdm(folders)
    for folder in pbar:
        # Find all ply files in each subdir
        file_path_list = sorted(glob(os.path.join(os.path.join(root, folder, "*.ply"))))
        assert(len(file_path_list) > 0)

        # Dict, to separate based on name
        data = {}

        for file_path in file_path_list:
            basename = os.path.basename(file_path)
            pbar.set_description(f"Processing {basename}")
            basename = basename[:-4]  # Hardcoded, remove .ply

            data_data = torch_geometric.io.read_ply(file_path) # Note, it is not raw, it is a data object

            if sample == "2pass": raise NotImplementedError()
            elif sample == "bruteforce":
                tri = torch_geometric.utils.to_trimesh(data_data)
                tri = reduction_transform.simplify_trimesh(tri, sample_size, 2048, 4096)
                data_sampled = torch_geometric.utils.from_trimesh(tri)
            elif sample == "random": raise NotImplementedError()
            elif sample == "all":
                data_sampled = data_data
            else:
                raise ValueError("Invalid argument", sample)
            
            data[basename] = data_sampled

        dataset[folder] = data
    return dataset


# DOES NOT CHECK IF FILTER IS CHANGED
# TODO maybe save entire dataset, followed by applying filter post?
# Or possibly both
def get_3dface_dict(root, pickled, force=False, picke_name="3dface_cache-reduced.p", sample="2pass", sample_size=2048):
    if pickled and not force:
        try:
            print("Loading pickle")
            dataset = pickle.load(open(picke_name, "rb"))
            print("Pickle loaded")
            return dataset
        except Exception as e:
            print(f"Pickle failed - {str(e)}, loading data manually")

    dataset = generate_3dface_dict(root, sample=sample, sample_size=sample_size)

    if pickled:
        print("Saving pickle")
        pickle.dump(dataset, open(picke_name, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickle saved")

    return dataset

