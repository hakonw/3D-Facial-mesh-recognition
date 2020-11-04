import os
import torch
from torch.utils.data import Dataset
from pytorch3d.io import load_obj

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class FaceGenDataset(Dataset):
    def __init__(self, device, root_dir="/lhome/haakowar/Downloads/FaceGen_DB/"):
        """
        Args:
            root_dir (string).
        """
        self.device = device
        self.root_dir = root_dir

        # TODO split into train val

        self.folders = []
        for root, dirs, filenames in os.walk(root_dir):
            self.folders = sorted(dirs)
            break  # prevent descending into subfolders
        print(self.folders)

        self.path_lookup = {}

        for short_path in self.folders:
            idx = int(short_path)
            long_path = os.path.join(self.root_dir, short_path)
            regular_obj_path = os.path.join(long_path, "Data", short_path + ".obj") # Get the regular object
            query_path = os.path.join(long_path, "Query")

            # Get the posed object
            query_obj_path = None
            for root, dirs, filenames in os.walk(query_path):
                for filename in filenames:
                    if ".obj" in filename:
                        query_obj_path = os.path.join(query_path, filename)
                        break
                break  # prevent descending into subfolders
            assert query_obj_path is not None

            sample_path = {"regular": regular_obj_path, "alt": query_obj_path, "idd": short_path}
            self.path_lookup[idx] = sample_path


    def __len__(self):
        return len(self.folders)


    # Etterligne https://pytorch3d.readthedocs.io/en/latest/modules/datasets.html  ?
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        paths = self.path_lookup[idx+1]  # +1 As the id's are labled from 1->100


        r_verts, r_faces, r_aux = load_obj(paths["regular"], load_textures=False, device=self.device)
        a_verts, a_faces, a_aux = load_obj(paths["alt"], load_textures=False, device=self.device)

        sample = {"regular":(r_verts, r_faces.verts_idx), "alt": (a_verts, a_faces.verts_idx), "idd": paths["idd"]}

        return sample



if __name__ == "__main__":
    # Testing
    dataset = FaceGenDataset(device="cpu")
    print("len", len(dataset))
    #print("idx0:", dataset[0])

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=4)

    for i_batch, sample_batced in enumerate(dataloader):
        print(i_batch, sample_batced["regular"][0].shape)

        from pytorch3d.structures import Meshes

        # mesh_face = Meshes(verts=[verts], faces=[faces.verts_idx])
        mesh = Meshes(verts=sample_batced["regular"][0], faces=sample_batced["regular"][1])

        # https://pytorch3d.readthedocs.io/en/latest/modules/structures.html
        # Alt, bruk join_meshes_as_batches


        aa = mesh.verts_packed()
        print(aa.shape)
        break
