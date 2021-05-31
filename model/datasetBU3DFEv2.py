import os.path
from glob import glob
import pickle
from tqdm import tqdm
from read_wrl import read_wrl
import read_bnt
import torch_geometric.utils
import reduction_transform

def global_relevant(name): return True

def generate_bu3dfe_dict(root, filtered=True, filter=global_relevant, sample="2pass", sample_size=2048):
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
        # Find all wrl files, and only seject the F3D.wrl files, and not the RAW.wrl files
        file_path_list = sorted(glob(os.path.join(os.path.join(root, folder, "*F3D.wrl"))))
        assert(len(file_path_list) > 0)

        # Dict, to separate based on name
        data = {}

        for file_path in file_path_list:
            basename = os.path.basename(file_path)
            pbar.set_description(f"Processing {basename}")
            basename = basename[:-8]  # Hardcoded, remove _F3D.wrl

            # Skip if the file is not relevant & the filter is on
            if filtered and not filter(basename):
                continue

            data_data = read_wrl(file_path)  # Note, it is not raw, it is a data object
            if sample == "2pass": raise NotImplementedError()
            elif sample == "bruteforce":
                tri = torch_geometric.utils.to_trimesh(data_data)
                tri = reduction_transform.simplify_trimesh(tri, sample_size, 2)
                data_sampled = torch_geometric.utils.from_trimesh(tri)
            elif sample == "random": raise NotImplementedError()
            elif sample == "all":
                data_sampled = data_data
            else: raise ValueError("Invalid argument", sample)
            data[basename] = data_sampled

        dataset[folder] = data
    return dataset


# DOES NOT CHECK IF FILTER IS CHANGED
# TODO maybe save entire dataset, followed by applying filter post?
# Or possibly both
def get_bu3dfe_dict(root, pickled, force=False, picke_name="BU-3DFE_cache-reduced.p", filter=global_relevant, sample="2pass", sample_size=2048):
    if pickled and not force:
        try:
            print("Loading pickle")
            dataset = pickle.load(open(picke_name, "rb"))
            print("Pickle loaded")
            return dataset
        except Exception as e:
            print(f"Pickle failed - {str(e)}, loading data manually")

    dataset = generate_bu3dfe_dict(root, filtered=True, filter=filter, sample=sample, sample_size=sample_size)

    if pickled:
        print("Saving pickle")
        pickle.dump(dataset, open(picke_name, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print("Pickle saved")

    return dataset

