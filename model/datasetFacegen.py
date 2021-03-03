import os.path
from glob import glob
import random
import pickle

# from torch_geometric.io import read_obj  # replaced with modified
from read_obj import read_obj
from utils import list_collate_fn
from torch.utils.data import Dataset
import torch_geometric.transforms

import torch_geometric.transforms as T

random.seed(1)
import torch
torch.manual_seed(1)

# In memory cache helper
class FaceGenDatasetHelper:
    _face_to_edge_transformator = torch_geometric.transforms.FaceToEdge(remove_faces=True)

    def __init__(self, root="/lhome/haakowar/Downloads/FaceGen_DB/", pickled=True, face_to_edge=True):
        self.dataset = {}

        if pickled:
            try:
                self.dataset = pickle.load(open("facegen_cache.p", "rb"))
                print("Pickle loaded")
                return
            except Exception as e:
                print(f"Pickle failed - {str(e)}, loading data manually")

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

            data = {}
            for file_path in file_paths:
                obj_file = read_obj(file_path)
                if face_to_edge:
                    obj_file = FaceGenDatasetHelper._face_to_edge_transformator(obj_file)  # Replace faces with edges
                basename = os.path.basename(file_path)[:-4]  # Get the filename, minus the extension (.obj)
                data[basename] = obj_file

            self.dataset[folder] = data

        if pickled:
            print("Saving pickle")
            pickle.dump(self.dataset, open("facegen_cache.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            print("Pickle saved")

    def get_cached_dataset(self):
        return self.dataset



# Possibly switch over to pytorch-gemoetric dataset
# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-transforms
# Eks: transform=T.RandomTranslate(0.01)
class FaceGenDataset(Dataset):
    def __init__(self, dataset_cache: dict):
        self.dataset_cache = dataset_cache
        self.dataset_keys = list(dataset_cache.keys())
        self.transform = T.Compose([T.ToUndirected()])  # T.Compose([T.NormalizeScale()])

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, idx):
        # Generate a pair of valid positive samples
        data = self.dataset_cache[self.dataset_keys[idx]]
        assert len(data) == 2
        # As there are only 2 scans for each identity, just return them both
        safe_dict = {n: self.transform(d.clone()) for n, d in data.items()}  # Make sure not to edit the originals
        return safe_dict
        # data1.__setitem__("id", idx)
        # data2.__setitem__("id", idx)
        # random_sample = random.sample(data, 2)  # get 2 random samples
        # assert random_sample[0] != random_sample[1]
        # return (random_sample[0].clone(), random_sample[1].clone())


# simple test
if __name__ == "__main__":
    facegen_helper = FaceGenDatasetHelper()
    dataset = FaceGenDataset(facegen_helper.get_cached_dataset())
    print("len", len(dataset))

    #import torch
    #dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, shuffle=False, num_workers=1)
    #from torch_geometric.data import DataLoader
    #import torch_geometric
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=list_collate_fn)

    # Single: Data(               face=[4, 5790],  pos=[5850, 3])
    # Double: Data(batch=[11700], face=[4, 11580], pos=[11700, 3])

    import torch_geometric
    for i_batch, sample_batced in enumerate(dataloader):
        print("pre", i_batch, sample_batced)
        print("length of batch", len(sample_batced))
        # print("sample list", sample_batced[0].to_data_list())
        # data = sample_batced[0].to_data_list()[0]
        # print("data", data)
        # print("data-face:", data.face)
        # print(torch_geometric.utils.geodesic_distance(data.pos, data.face))
        # face_to_edge = torch_geometric.transforms.FaceToEdge(remove_faces=True)
        # print("edge data", face_to_edge(data))

        break