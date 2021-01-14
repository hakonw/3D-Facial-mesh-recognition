import os.path
from glob import glob

import torch
from torch_geometric.data import InMemoryDataset
# from torch_geometric.io import read_obj  # replaced with modified
from read_obj import read_obj
from torch_geometric.data import Data, DataLoader



class FaceGen(InMemoryDataset):
    def __init__(self, root="/lhome/haakowar/Downloads/FaceGen_DB/"):
        super(FaceGen, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root)

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        folders = []
        for root, dirs, filenames in os.walk(self.raw_dir):
            folders = sorted(dirs)
            break  # prevent descending into subfolders
        if "processed" in folders:
            folders.remove("processed")
        
        data_list = []
        for folder in folders:
            # Get the regular face
            file_path = os.path.join(self.raw_dir, folder, "Data", folder + ".obj")
            # print(file_path)
            data_regular = read_obj(file_path)
            # print(data_regular)

            # Get the alternative identity
            file_path = sorted(glob(os.path.join(os.path.join(self.raw_dir, folder, "Query", "*.obj"))))
            assert len(file_path) == 1
            data_alternative = read_obj(file_path[0])
            data = Data(faces2=[data_regular, data_alternative], id=folder)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


if __name__ == "__main__":
    print("Beginning main")
    dataset = FaceGen()
    dataloader = DataLoader(dataset=dataset, batch_size=25, shuffle=True, num_workers=4)
    for i_batch, sample_batced in enumerate(dataloader):
        print(sample_batced)