import os.path
from glob import glob
import torch
import pickle
from tqdm import tqdm

#from torch.utils.data import Dataset
from torch_geometric.data import Dataset

import torch_geometric.transforms
import torch_geometric.transforms as T

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

    def __init__(self, root, pickled=True, face_to_edge=False, device="cpu"):
        self.dataset = {}

        if pickled:
            try:
                self.dataset = pickle.load(open("BU-3DFE_cache.p", "rb"))
                import sys
                print("Pickle loaded")
                transform = T.NormalizeScale()
                # device = torch.device("cuda:0")

                with torch.no_grad():
                    for face_id in self.dataset.keys():
                        for scan in self.dataset[face_id].keys():
                            # self.dataset[face_id][scan] = transform(self.dataset[face_id][scan])
                            # self.dataset[face_id][scan].pos = torch.div(self.dataset[face_id][scan].pos, 100).to(device)
                            # print(sys.getsizeof(self.dataset[face_id][scan].pos.storage()))
                            pass
                # import time
                # print("done")
                # time.sleep(5)
                return
            except Exception as e:
                print(f"Pickle failed - {str(e)}, loading data manually")
                assert 1 > 3

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

                data_file = read_wrl(file_path).to(device)
                if face_to_edge:
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
    def __init__(self, dataset_cache: dict, posttransform, name_filter=None):
        self.dataset_cache = dataset_cache
        self.dataset_keys = list(dataset_cache.keys())
        self.transform = posttransform
        self.filter = name_filter
        if self.filter is None:
            self.filter = lambda l: l == "00" or l == "01" or l == "02"
            # Alt self.filter = lambda l: True

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, idx):
        data = self.dataset_cache[self.dataset_keys[idx]]
        # Data is a new dict
        safe_dict = {}
        for name, d in data.items():
            level = name[8:10]
            if self.filter(level): # if level == "01" or level == "00" or level == "02":

                # import torch_geometric.io
                # torch_geometric.io.write_off(d, f"./{name}.off")
                # Gather some statistics about the first graph.
                # print(f'Number of nodes: {d.num_nodes}')
                # print(f'Number of edges: {d.num_edges}')
                # print(f'Average node degree: {d.num_edges / d.num_nodes:.2f}')
                # print(f'Contains isolated nodes: {d.contains_isolated_nodes()}')
                # print(f'Contains self-loops: {d.contains_self_loops()}')
                # print(f'Is undirected: {d.is_undirected()}')

                safe_dict[name] = self.transform(d.clone())  # Make sure not to edit the originals

        # Transform into pytorch dataset
        out = []
        for name, d in safe_dict.items():
            d.id = idx
            d.name = name
            out.append(d)
        return out

        # TODO generate some filter somhow
        return safe_dict


if __name__ == "__main__2":
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


if __name__ == "__main__2":
    import torch
    from torch_geometric.data import DataLoader
    import torch_geometric.transforms as T
    import torch_geometric.io

    import sys

    POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True), T.NormalizeScale()])
    torch.manual_seed(1); torch.cuda.manual_seed(1)    

    DATASET_PATH_BU3DFE = "/lhome/haakowar/Downloads/BU_3DFE/"
    BU3DFE_HELPER = BU3DFEDatasetHelper(root=DATASET_PATH_BU3DFE, pickled=True, face_to_edge=False)
    dataset_cached = BU3DFE_HELPER.get_cached_dataset()

    # Load dataset and split into train/test 
    dataset = BU3DFEDataset(dataset_cached, POST_TRANSFORM, name_filter=lambda l: True)

    # Regular dataloader followed by two test dataloader (seen data, and unseen data)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    
    import torch_geometric.nn as nn
    from torch_geometric.data import Data

    with torch.no_grad():
        for major_batch in dataloader:
            for minor_batch in major_batch:
                print(minor_batch)
                data = minor_batch.get_example(0)
                print(data)

                name = data.name
                data_org = data

                # if False:
                # print("ASA")
                # data = data_org.clone()
                # pooling = nn.ASAPooling(in_channels=3, ratio=512)
                # x, edge_index, edge_weight, batch, perm = pooling(x=data.pos, edge_index=data.edge_index)
                # data_pool1 = Data(x=None, edge_index=edge_index, pos=x, face=torch.empty(3, 0, dtype=torch.long))
                # torch_geometric.io.write_off(data_pool1, f"./{name}-ASAPool.off")
            

                # # Edge pooling renger en edge score metric , usikker hva å gå 

                # print("mem")
                # data = data_org.clone()
                # pooling = nn.MemPooling(in_channels=3, out_channels=3, heads=24, num_clusters=24)
                # x, s = pooling(x=data.pos)
                # data_pool2 = Data(x=None, edge_index=edge_index, pos=x, face=torch.empty(3, 0, dtype=torch.long))
                # torch_geometric.io.write_off(data_pool2, f"./{name}-MemPool.off")


                # print("pan")
                # data = data_org.clone()
                # pooling = nn.PANPooling(in_channels=3, ratio=0.5)
                # transf = T.ToSparseTensor()
                # data.name=None
                # data_prepool3 = transf(data)
                # print(data_prepool3)
                # x, edge_index, edge_attr, batch, perm, score = pooling(data_prepool3.pos, data_prepool3.adj_t)
                # data_pool3 = Data(x=None, edge_index=edge_index, pos=x, face=torch.empty(3, 0, dtype=torch.long))
                # torch_geometric.io.write_off(data_pool3, f"./{name}-PANPool.off")


                print("sag")
                data = data_org.clone()
                pooling = nn.SAGPooling(in_channels=3, ratio=1024*6)
                x, edge_index, edge_attr, batch, perm, score = pooling(x=data.pos, edge_index=data.edge_index, edge_attr=data.edge_attr)
                data_pool4 = Data(x=None, edge_index=edge_index, pos=x, face=torch.empty(3, 0, dtype=torch.long))
                torch_geometric.io.write_off(data_pool4, f"./{name}-SAGPooling-6k.off")


                # print("topk")
                # data = data_org.clone()
                # pooling = nn.TopKPooling(in_channels=3, ratio=512)
                # x, edge_index, edge_attr, batch, perm, score = pooling(x=data.pos, edge_index=data.edge_index, edge_attr=data.edge_attr)
                # data_pool5 = Data(x=None, edge_index=edge_index, pos=x, face=torch.empty(3, 0, dtype=torch.long))
                # torch_geometric.io.write_off(data_pool5, f"./{name}-TopKPooling.off")


                # avg_pool trenger et cluster. mulig å bruke?

                # avg_pool_neighbor_x gir ikke mening? tar avg av x (neighbors)

                # avg_pool_x gir ikke mening, må ha clusters for avg x

                # fps, samples most distant point regards to the rest
                # gir indexes 

                # graclus, idk

                # knn, lager clusters (indexies) av knn da i guess, mulig å bruke med noe annet?

                # gnn_graph:  lager en knn graph, så den kobler alle knn sammen

                # max_pool, "pools and corsens" based på cluster, all nodes inside same cluster will become 1 node 

                # max_pool_neighbor_x, tar x fra naboen

                # max_pool_x, tar x basert på cluster

                # nearest, clustrer x basert på y

                # radius, for alle elementer i y, finn alle x av radius r

                # radius_graph, lag en graph basert på radius, aka koble alle som er r nære hverandre sammen

                # voxel_grid, ting

                # data.face = torch.empty(3, 0)
                # torch_geometric.io.write_off(data, f"./{name}-KNN-org.off")

                sys.exit(0)


if __name__ == "__main__":
    import trimesh
    import torch_geometric
    from torch_geometric.data import DataLoader

    import sys

    # POST_TRANSFORM = T.Compose([T.FaceToEdge(remove_faces=True), T.NormalizeScale()])
    POST_TRANSFORM = T.Compose([T.NormalizeScale()])
    torch.manual_seed(1); torch.cuda.manual_seed(1)    

    DATASET_PATH_BU3DFE = "/lhome/haakowar/Downloads/BU_3DFE/"
    BU3DFE_HELPER = BU3DFEDatasetHelper(root=DATASET_PATH_BU3DFE, pickled=True, face_to_edge=False)
    dataset_cached = BU3DFE_HELPER.get_cached_dataset()

    # Load dataset and split into train/test 
    dataset = BU3DFEDataset(dataset_cached, POST_TRANSFORM, name_filter=lambda l: True)

    # Regular dataloader followed by two test dataloader (seen data, and unseen data)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    
    import tqdm
    for major_batch in tqdm.tqdm(dataloader):
        for minor_batch in major_batch:
            # print("minor: ", minor_batch)
            data = minor_batch.get_example(0)
            # print("data: ", data)

            trimesh = torch_geometric.utils.to_trimesh(data)
            # print("trimesh: ", trimesh)
            vertices = trimesh.vertices.shape[0]
            faces = trimesh.faces.shape[0]
            # print("verticices ", vertices)
            # print("faces ", faces)

            # for each 2 reduction in faces. 1 verticices is removed

            must_remove_vertices = vertices - 2048
            must_remove_faces = must_remove_vertices

            tmpmesh = trimesh
            total = 0 
            for i in range(50):
                vertices = tmpmesh.vertices.shape[0]
                faces = tmpmesh.faces.shape[0]
                must_remove_vertices = vertices - 2048
                must_remove_faces = must_remove_vertices

                if must_remove_vertices == 0:
                    # print(True)
                    break
                if must_remove_vertices < 0:
                    print(False)
                    break

                tmpmesh = tmpmesh.simplify_quadratic_decimation(faces - must_remove_faces)
                total += 1
                
                # if i == 0:
                #     tmpmesh_2x = tmpmesh.simplify_quadratic_decimation(faces - must_remove_faces)
                #     total += 1
                # else: 
                #     tmpmesh_2x = tmpmesh.simplify_quadratic_decimation(faces - must_remove_faces*1.2)
                #     total += 1
                # if tmpmesh_2x.vertices.shape[0] < 2048:
                #     tmpmesh = tmpmesh.simplify_quadratic_decimation(faces - must_remove_faces)
                #     total += 1
                # else:
                #     tmpmesh = tmpmesh_2x
            print(total)
                

            # trimesh_simplified.export(f"./{data.name}-reduced.ply")

            import sys 
        # sys.exit(0)