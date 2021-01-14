import os.path
from glob import glob

# from torch_geometric.io import read_obj  # replaced with modified
from read_obj import read_obj
from torch.utils.data import Dataset
#import numpy as np
import random
import pickle

# In memory cache helper
class FaceGenDatasetHelper:
    def __init__(self, root="/lhome/haakowar/Downloads/FaceGen_DB/", pickled=True):
        self.dataset = {}

        if pickled:
            try:
                self.dataset = pickle.load(open("facegen_cache.p", "rb"))
                print("Pickle loaded")
                return
            except:
                print("Pickle failed, loading data manually")
                pass

        # Find all sub-folders
        folders = []
        for root, dirs, filenames in os.walk(root):
            folders = sorted(dirs)
            break  # prevent descending into subfolders

        # Load the model for each identity
        for folder in folders:
            file_path_reg = os.path.join(root, folder, "Data", folder + ".obj")
            file_path_alt_list = sorted(glob(os.path.join(os.path.join(root, folder, "Query", "*.obj"))))
            assert len(file_path_alt_list) == 1  # Mostly a precaution to see more about the data
            file_paths = [file_path_reg] + file_path_alt_list

            data = []
            for file_path in file_paths:
                data.append(read_obj(file_path))

            self.dataset[folder] = data

        if pickled:
            print("Saving pickle")
            pickle.dump(self.dataset, open("facegen_cache.p", "wb"), protocol=-1)
            print("Pickle saved")

    def get_cached_dataset(self):
        return self.dataset


class FaceGenDataset(Dataset):
    def __init__(self, dataset_cache: dict):
        self.dataset_cache = dataset_cache
        self.dataset_keys = list(dataset_cache.keys())

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, idx):
        # Generate a pair of valid positive samples
        data = self.dataset_cache[self.dataset_keys[idx]]
        assert len(data) >= 2  # Impossible otherwise, need to change strategy then
        random_sample = random.sample(data, 2)  # get 2 random samples
        assert random_sample[0] != random_sample[1]
        #return {"a": random_sample[0], "b": random_sample[1]}
        return random_sample


# simple test
if __name__ == "__main__":
    facegen_helper = FaceGenDatasetHelper()
    dataset = FaceGenDataset(facegen_helper.get_cached_dataset())
    print("len", len(dataset))

    #import torch
    #dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, shuffle=False, num_workers=1)
    from torch_geometric.data import DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, num_workers=1)

    # Single: Data(               face=[4, 5790],  pos=[5850, 3])
    # Double: Data(batch=[11700], face=[4, 11580], pos=[11700, 3])

    import torch_geometric
    for i_batch, sample_batced in enumerate(dataloader):
        print(i_batch, sample_batced)
        print(len(sample_batced))
        print(sample_batced[0].to_data_list())
        #torch_geometric.utils.to_dense_adj(sample_batced)
        break