import os.path
from glob import glob
import pickle
from tqdm import tqdm
from meshfr.io.read_wrl import read_wrl
import torch_geometric.utils
from . import reduction_transform
from multiprocessing import Pool
import os

def global_relevant(name): return True

class Process_single(object):
    def __init__(self, sample, sample_size):
        self.sample = sample 
        self.sample_size = sample_size
    def __call__(self, file_path):
        data_data = read_wrl(file_path)  # Note, it is not raw, it is a data object
        if self.sample == "2pass": raise NotImplementedError()
        elif self.sample == "bruteforce":
            tri = torch_geometric.utils.to_trimesh(data_data)
            tri = reduction_transform.simplify_trimesh(tri, self.sample_size, self.sample_size//32, self.sample_size//32)
            data_sampled = torch_geometric.utils.from_trimesh(tri)
        elif self.sample == "random": raise NotImplementedError()
        elif self.sample == "all":
            data_sampled = data_data
        else: raise ValueError("Invalid argument", self.sample)
        # data[basename] = data_sampled
        return data_sampled


def generate_bu3dfe_dict(root, filtered=True, filter=global_relevant, sample="2pass", sample_size=None, multithread=False):
    assert sample_size
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

    assert len(folders) > 0, f"No identity found in folder {root}"
    # Find all scans for each identity
    pbar = tqdm(folders)
    if multithread:
        pool = Pool(os.cpu_count()//2)  # Multi processing as i got tired of waiting
    process_single = Process_single(sample, sample_size)
    for folder in pbar:
        # Find all wrl files, and only seject the F3D.wrl files, and not the RAW.wrl files
        file_path_list = sorted(glob(os.path.join(os.path.join(root, folder, "*F3D.wrl"))))
        assert(len(file_path_list) > 0)
        pbar.set_description(f"Processing {folder}")

        # Dict, to separate based on name
        data = {}
        inputs = []
        basenames = []
        for file_path in file_path_list:
            basename = os.path.basename(file_path)
            basename = basename[:-8]  # Hardcoded, remove _F3D.wrl
            if filtered and not filter(basename): # Skip if the file is not relevant & the filter is on
                return
            inputs.append(file_path)
            basenames.append(basename)
        if multithread:
            results = pool.map(process_single, inputs)
        else:
            results = list(map(process_single, inputs))

        for i in range(len(results)):
            data[basenames[i]] = results[i]
            

        dataset[folder] = data
    if multithread:
        pool.close()
        pool.join()
    return dataset


# DOES NOT CHECK IF FILTER IS CHANGED
# TODO maybe save entire dataset, followed by applying filter post?
# Or possibly both
def get_bu3dfe_dict(root, pickled, force=False, picke_name="BU-3DFE_cache-reduced.p", filter=global_relevant, sample="2pass", sample_size=None):
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

