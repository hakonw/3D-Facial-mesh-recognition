import os.path
from glob import glob
import torch
import pickle
from tqdm import tqdm

from torch.utils.data import Dataset
import torch_geometric.transforms

from read_wrl import read_wrl
from utils import list_collate_fn

torch.manual_seed(1)


# Format:
# F/M ID: F0001, M0001   female or man,
#   id _ [AN/DI/FE/HA/NE/SA/SU] ?? [00-04] [F3D/RAW] . [bnd/wrl/pse]
#     It is about that, raw will have .wrl or .pse,  F3d will have .wrl or .bnd
#     Not all emotions have 4 states, neutral only has 00, the others have 01-04
#     TODO look into ??,  have been wh, ae, la, bl, am, in
#   Example:  F0001_AN03WH_F3D.wrl   F0001_NE00WH_F3D.wrl  F0001_SA04WH_F3D.wrl
#
#  Stands for:  Angry, disgust, fe?, happy, neutral, sad, su? TODO


class BU3DFEDatasetHelper:
    # classes = ["AN", "DI", "FE", "HA", "NE", "SA", "SU"]
    _face_to_edge_transformator = torch_geometric.transforms.FaceToEdge(remove_faces=True)

    def __init__(self, root, pickled=True):
        self.dataset = {}

        if pickled:
            try:
                self.dataset = pickle.load(open("BU-3DFE_cache.p", "rb"))
                print("Pickle loaded")
                return
            except Exception as e:
                print(f"Pickle failed - {str(e)}, loading data manually")

        print("This will take some time")

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
            # Could also split into classes, then into numbers based on ids
            # TODO find best setup
            data = {}

            for file_path in file_path_list:
                basename = os.path.basename(file_path)
                pbar.set_description(f"Processing {basename}")
                basename = basename[:-8]  # Hardcoded, remove _F3D.wrl

                data_file = read_wrl(file_path)
                data_file = BU3DFEDatasetHelper._face_to_edge_transformator(data_file)  # Replace faces with edges

                data[basename] = data_file

            self.dataset[folder] = data

        if pickled:
            print("Saving pickle")
            pickle.dump(self.dataset, open("BU-3DFE_cache.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            print("Pickle saved")

    def get_cached_dataset(self):
        return self.dataset


class BU3DFEDataset(Dataset):  # This does not need to be of type Dataset
    def __init__(self, dataset_cache: dict):
        self.dataset_cache = dataset_cache
        self.dataset_keys = list(dataset_cache.keys())

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, idx):
        data = self.dataset_cache[self.dataset_keys[idx]]
        # Data is a new dict
        safe_dict = {}
        for name, d in data.items():
            safe_dict[name] = d.clone()  # Make sure not to edit the originals

        # TODO generate some filter somhow

        return safe_dict


if __name__ == "__main__":
    path = "/lhome/haakowar/Downloads/BU_3DFE/"
    BU3DFE_helper = BU3DFEDatasetHelper(path)

    dataset = BU3DFEDataset(BU3DFE_helper.get_cached_dataset())

    # Currently, pytorch_geometric overwrites the collate_fn
    # https://github.com/rusty1s/pytorch_geometric/blob/3e8baf28c86eebbf6da74be36ea3904ec77480b8/torch_geometric/data/dataloader.py#L57
    # So that's a thing

    # from torch_geometric.data import DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, num_workers=1, collate_fn=list_collate_fn)

    for i_batch, sample_batced in enumerate(dataloader):
        print("pre", i_batch, sample_batced)
        print("length of batch", len(sample_batced))
        # print("sample list", sample_batced[0].to_data_list())
        # data = sample_batced[0].to_data_list()[0]
        # print("data", data)
        # print("data-face:", data.face)
        break
